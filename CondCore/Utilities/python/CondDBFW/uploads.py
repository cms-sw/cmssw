#!/usr/bin/env python 
"""

Joshua Dawes - CERN, CMS - The University of Manchester

This module holds classes to help with uploading conditions to the drop box web service, which also uses CondDBFW to read and write data.

"""
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import base64
from datetime import datetime
from urllib import urlencode
import math
import sys
import traceback
import netrc

from .url_query import url_query
from . import models
from . import errors
from . import data_sources
from . import querying
from .errors import *
from .utils import to_timestamp, to_datetime, friendly_since

def friendly_since(time_type, since):
    """
    Takes a since and, if it is Run-based expressed as Lumi-based, returns the run number.
    Otherwise, returns the since without transformations.
    """
    if time_type == "Run" and (since & 0xffffff) == 0:
        return since >> 32
    else:
        return since

# this is simple, and works for now - if logging requirements change, I will write a logging class to manage logging
def log(file_handle, message):
	"""
	Very simple logging function, used by output class.
	"""
	file_handle.write("[%s] %s\n" % (to_timestamp(datetime.now()), message))

def new_log_file_id():
	"""
	Find a new client-side log file name.

	Note: This cannot use the upload session token since logs need to be written before this is opened.
	However, this can be changed so that the filename that uses the token is written to once
	it is obtained.
	"""
	# new id = number of log files + 1
	# (primitive - matching the hash of the upload session may be a better idea)
	log_files = [file for file in os.listdir(os.path.join(os.getcwd(), "upload_logs")) if "upload_log" in file]
	new_id = len(log_files)+1
	return new_id

class output():
	INFO = 0
	ERROR = 1
	WARNING = 2
	VERBOSE = 3
	DEBUG = 4
	
	"""
	Used to control output to the console and to the client-side log.
	"""

	def __init__(self, log_handle=None, verbose=False, debug=False):
		# first time writing progress bar, don't need to go back along the line
		self.current_output_length = 0
		self._verbose = verbose
		self._log_handle = log_handle
		self._debug = debug
		self.labels = ["INFO", "ERROR", "WARNING", "VERBOSE", "DEBUG"]

	def write(self, message="", level=INFO):
		"""
		Write to the console and to the log file held by self.
		"""
		message = "[%s] %s: %s"%(datetime.now(), self.labels[level], message)
		if self._verbose:
			if level == output.DEBUG and self._debug:
				print(message)
			elif level < output.DEBUG:
				print(message)
		elif self._debug:
			if level == output.DEBUG:
				print(message)
		elif level <= output.ERROR:
			print(message)
		if self._log_handle != None:
			log(self._log_handle, message)

class uploader(object):
	"""
	Upload session controller - creates, tracks, and deletes upload sessions on the server.
	"""

	def __init__(self, metadata_source=None, debug=False, verbose=False, testing=False, server="https://cms-conddb-dev.cern.ch/cmsDbCondUpload/", **kwargs):
		"""
		Upload constructor:
		Given an SQLite file and a Metadata sources, reads into a dictionary read for it to be encoded and uploaded.

		Note: kwargs is used to capture stray arguments - arguments that do not match keywords will not be used.

		Note: default value of service_url should be changed for production.
		"""
		# set private variables
		self._debug = debug
		self._verbose = verbose
		self._testing = testing
		# initialise server-side log data as empty string - will be replaced when we get a response back from the server
		self._log_data = ""
		self._SERVICE_URL = server
		self.upload_session_id = None

		# set up client-side log file
		self.upload_log_file_name = "upload_logs/upload_log_%d" % new_log_file_id()
		self._handle = open(self.upload_log_file_name, "a")

		# set up client-side logging object
		self._outputter = output(verbose=verbose, log_handle=self._handle, debug = self._debug)
		self._outputter.write("Using server instance at '%s'." % self._SERVICE_URL)

		# expect a CondDBFW data_source object for metadata_source
		if metadata_source == None:
			# no upload metadat has been given - we cannot continue with the upload
			self.exit_upload("A source of metadata must be given so CondDBFW knows how to upload conditions.")
		else:
			# set up global metadata source variable
			self.metadata_source = metadata_source.data()

		# check for the destination tag
		# this is required whatever type of upload we're performing
		if self.metadata_source.get("destinationTags") == None:
			self.exit_upload("No destination Tag was given.")
		else:
			if isinstance(self.metadata_source.get("destinationTags"), dict) and self.metadata_source.get("destinationTags").keys()[0] == None:
				self.exit_upload("No destination Tag was given.")

		# make sure a destination database was given
		if self.metadata_source.get("destinationDatabase") == None:
			self.exit_upload("No destination database was given.")

		# get Conditions metadata
		if self.metadata_source.get("sourceDB") == None and self.metadata_source.get("hashToUse") == None:
			"""
			If we have neither an sqlite file nor the command line data
			"""
			self.exit_upload("You must give either an SQLite database file, or the necessary command line arguments to replace one."\
							+ "\nSee --help for command line argument information.")
		elif self.metadata_source.get("sourceDB") != None:
			"""
			We've been given an SQLite file, so try to extract Conditions Metadata based on that and the Upload Metadata in metadata_source
			We now extract the Tag and IOV data from SQLite.  It is added to the dictionary for sending over HTTPs later.
			"""

			# make sure we have an input tag to look for in the source db
			self.input_tag = metadata_source.data().get("inputTag")
			if self.input_tag == None:
				self.exit_upload("No input Tag name was given.")

			# set empty dictionary to contain Tag and IOV data from SQLite
			result_dictionary = {}
			self.sqlite_file_name = self.metadata_source["sourceDB"]
			if not(os.path.isfile(self.sqlite_file_name)):
				self.exit_upload("SQLite file '%s' given doesn't exist." % self.sqlite_file_name)
			sqlite_con = querying.connect("sqlite://%s" % os.path.abspath(self.sqlite_file_name))

			self._outputter.write("Getting Tag and IOVs from SQLite database.", output.VERBOSE)

			# query for Tag, check for existence, then convert to dictionary
			tag = sqlite_con.tag(name=self.input_tag)
			if tag == None:
				self.exit_upload("The source Tag '%s' you gave was not found in the SQLite file." % self.input_tag)
			tag = tag.as_dicts(convert_timestamps=True)

			# query for IOVs, check for existence, then convert to dictionaries
			iovs = sqlite_con.iov(tag_name=self.input_tag)
			if iovs == None:
				self.exit_upload("No IOVs found in the SQLite file given for Tag '%s'." % self.input_tag)
			iovs = iovs.as_dicts(convert_timestamps=True)
			iovs = [iovs] if not isinstance(iovs, list) else iovs

			"""
			Finally, get the list of all Payload hashes of IOVs,
			then compute the list of hashes for which there is no Payload for
			this is used later to decide if we can continue the upload if the Payload was not found on the server.
			"""
			iovs_for_hashes = sqlite_con.iov(tag_name=self.input_tag)
			if iovs_for_hashes.__class__ == data_sources.json_list:
				hashes_of_iovs = iovs_for_hashes.get_members("payload_hash").data()
			else:
				hashes_of_iovs = [iovs_for_hashes.payload_hash]
			self.hashes_with_no_local_payload = [payload_hash for payload_hash in hashes_of_iovs if sqlite_con.payload(hash=payload_hash) == None]

			# close session open on SQLite database file
			sqlite_con.close_session()

		elif metadata_source.data().get("hashToUse") != None:
			"""
			Assume we've been given metadata in the command line (since no sqlite file is there, and we have command line arguments).
			We now use Tag and IOV data from command line.  It is added to the dictionary for sending over HTTPs later.
			"""

			# set empty dictionary to contain Tag and IOV data from command line
			result_dictionary = {}

			now = to_timestamp(datetime.now())
			# tag dictionary will be taken from the server
			# this does not require any authentication
			tag = self.get_tag_dictionary()
			self.check_response_for_error_key(tag)
			iovs = [{"tag_name" : self.metadata_source["destinationTag"], "since" : self.metadata_source["since"], "payload_hash" : self.metadata_source["hashToUse"],\
					"insertion_time" : now}]

			# hashToUse cannot be stored locally (no sqlite file is given), so register it as not found
			self.hashes_with_no_local_payload = [self.metadata_source["hashToUse"]]

			# Note: normal optimisations will still take place - since the hash checking stage can tell if hashToUse does not exist on the server side

		# if the source Tag is run-based, convert sinces to lumi-based sinces with lumi-section = 0
		if tag["time_type"] == "Run":
			for (i, iov) in enumerate(iovs):
				iovs[i]["since"] = iovs[i]["since"] << 32

		result_dictionary = {"inputTagData" : tag, "iovs" : iovs}

		# add command line arguments to dictionary
		# remembering that metadata_source is a json_dict object
		result_dictionary.update(metadata_source.data())

		# store in instance variable
		self.data_to_send = result_dictionary

		# if the since doesn't exist, take the first since from the list of IOVs
		if result_dictionary.get("since") == None:
			result_dictionary["since"] = sorted(iovs, key=lambda iov : iov["since"])[0]["since"]
		elif self.data_to_send["inputTagData"]["time_type"] == "Run":
			# Tag time_type says IOVs use Runs for sinces, so we convert to Lumi-based for uniform processing
			self.data_to_send["since"] = self.data_to_send["since"] << 32

	@check_response(check="json")
	def get_tag_dictionary(self):
		url_data = {"tag_name" : self.metadata_source["destinationTag"], "database" : self.metadata_source["destinationDatabase"]}
		request = url_query(url=self._SERVICE_URL + "get_tag_dictionary/", url_data=url_data)
		response = request.send()
		return response

	def check_response_for_error_key(self, response_dict, exit_if_error=True):
		"""
		Checks the decoded response of an HTTP request to the server.
		If it is a dictionary, and one of its keys is "error", the server returned an error
		"""
		# if the decoded response data is a dictionary and has an error key in it, we should display an error and its traceback
		if isinstance(response_dict, dict) and "error" in response_dict.keys():
			splitter_string = "\n%s\n" % ("-"*50)
			self._outputter.write("\nERROR: %s" % splitter_string, output.ERROR)
			self._outputter.write(response_dict["error"], output.ERROR)

			# if the user has given the --debug flag, show the traceback as well
			if self._debug:
				# suggest to the user to email this to db upload experts
				self._outputter.write("\nTRACEBACK (since --debug is set):%s" % splitter_string, output.DEBUG)
				if response_dict.get("traceback") != None:
					self._outputter.write(response_dict["traceback"], output.DEBUG)
				else:
					self._outputter.write("No traceback was returned from the server.", output.DEBUG)
			else:
				self._outputter.write("Use the --debug option to show the traceback of this error.", output.INFO)

			# write server side log to client side (if we have an error from creating an upload session, the log is in its initial state (""))
			# if an error has occurred on the server side, a log will have been written
			self.write_server_side_log(response_dict.get("log_data"))

			if exit_if_error:
				if self._testing:
					return False
				else:
					exit()
		elif not("error" in response_dict.keys()) and "log_data" in response_dict.keys():
			# store the log data, if it's there, in memory - this is used if a request times out and we don't get any log data back
			self._log_data = response_dict["log_data"][2:-1]
			return True

	def write_server_side_log(self, log_data):
		"""
		Given the log data from the server, write it to a client-side log file.
		"""
		# if the server_side_log directory doesn't exist, create it
		# without it we can't write the log when we download it from the server
		if not(os.path.exists(os.path.join(os.getcwd(), "server_side_logs/"))):
			os.makedirs("server_side_logs/")

		# directory exists now, write to client-side log file
		server_log_file_name = None
		try:
			# if the upload session does not exist yet, don't try to write the log file
			if self.upload_session_id == None:
				raise Exception("No upload session")
			# create a write handle to the file, decode the log data from base64, write and close
			server_log_file_name = "server_side_logs/upload_log_%s" % str(self.upload_session_id)
			handle = open(server_log_file_name, "w")
			handle.write(base64.b64decode(log_data))
			handle.close()
		except Exception as e:
			# reset log file name to None so we don't try to write it later
			server_log_file_name = None
			#self._outputter.write("Couldn't write the server-side log file.\nThis may be because no upload session could be opened.")

		# tell the user where the log files are
		# in the next iteration we may just merge the log files and store one log (how it's done in the plotter module)
		if self._SERVICE_URL.startswith("https://cms-conddb-dev.cern.ch/cmsDbCondUpload"):
			logUrl = "https://cms-conddb.cern.ch/cmsDbBrowser/logs/show_cond_uploader_log/Prep/%s"%self.upload_session_id
		else:
			logUrl = "https://cms-conddb.cern.ch/cmsDbBrowser/logs/show_cond_uploader_log/Prod/%s"%self.upload_session_id
		
		print("[%s] INFO: Server log found at %s." % (datetime.now(), logUrl))
		if server_log_file_name != None:
			print("[%s] INFO: Local copy of server log file at '%s'." % (datetime.now(), server_log_file_name))
		else:
			print("No server log file could be written locally.")

		print("[%s] INFO: Local copy of client log file at '%s'." % (datetime.now(), self.upload_log_file_name))

	def exit_upload(self, message=None):
		"""
		Used to exit the script - which only happens if an error has occurred.
		If the --testing flag was passed by the user, we should return False for failure, and not exit
		"""
		if self.upload_session_id != None:
			# only try to close the upload session if an upload session has been obtained
			response = self.close_upload_session(self.upload_session_id)
			no_error = self.check_response_for_error_key(response)
			# if no error was found in the upload session closure request,
			# we still have to write the server side log
			if no_error:
				self.write_server_side_log(self._log_data)
		# close client-side log handle
		self._handle.close()
		if message != None:
			print("\n%s\n" % message)
		if self._testing:
			return False
		else:
			exit()

	def upload(self):
		"""
		Calls methods that send HTTP requests to the upload server.
		"""

		"""
		Open an upload session on the server - this also gives us a tag lock on the tag being uploaded, if it is available.
		"""
		try:

			# get upload session, check response for error key
			upload_session_data = self.get_upload_session_id()
			no_error = self.check_response_for_error_key(upload_session_data)

			# if there was an error and we're testing, return False for the testing module
			if not(no_error) and self._testing:
				return False

			self.upload_session_id = upload_session_data["id"]
			self._outputter.write("Upload session obtained with token '%s'." % self.upload_session_id, output.DEBUG)
			self.server_side_log_file = upload_session_data["log_file"]

		except errors.NoMoreRetriesException as no_more_retries:
			return self.exit_upload("Ran out of retries opening an upload session, where the limit was 3.")
		except Exception as e:
			# something went wrong that we have no specific exception for, so just exit and output the traceback if --debug is set.
			self._outputter.write(traceback.format_exc(), output.ERROR)

			if not(self._verbose):
				self._outputter.write("Something went wrong that isn't handled by code - to get the traceback, run again with --verbose.")
			else:
				self._outputter.write("Something went wrong that isn't handled by code - the traceback is above.")

			return self.exit_upload()

		"""
		Only if a value is given for --fcsr-filter, run FCSR filtering on the IOVs locally.
		"""
		if self.data_to_send["fcsr_filter"] != None:
			"""
			FCSR Filtering:
			Filtering the IOVs before we send them by getting the First Conditions Safe Run
			from the server based on the target synchronization type.
			"""
			if self.data_to_send["inputTagData"]["time_type"] != "Time":
				# if we have a time-based tag, we can't do FCSR validation - this is also the case on the server side
				try:
					self.filter_iovs_by_fcsr(self.upload_session_id)
					# this function does not return a value, since it just operates on data - so no point checking for an error key
					# the error key check is done inside the function on the response from the server
				except errors.NoMoreRetriesException as no_more_retries:
					return self.exit_upload("Ran out of retries trying to filter IOVs by FCSR from server, where the limit was 3.")
				except Exception as e:
					# something went wrong that we have no specific exception for, so just exit and output the traceback if --debug is set.
					self._outputter.write(traceback.format_exc(), output.ERROR)

					if not(self._verbose):
						self._outputter.write("Something went wrong that isn't handled by code - to get the traceback, run again with --verbose.")
					else:
						self._outputter.write("Something went wrong that isn't handled by code - the traceback is above.")

					return self.exit_upload()
			else:
				self._outputter.write("The Tag you're uploading is time-based, so we can't do any FCSR-based validation.  FCSR filtering is being skipped.")

		"""
		Check for the hashes that the server doesn't have - only send these (but in the next step).
		"""
		try:

			check_hashes_response = self.get_hashes_to_send(self.upload_session_id)
			# check for an error key in the response
			no_error = self.check_response_for_error_key(check_hashes_response)

			# if there was an error and we're testing, return False for the testing module
			if not(no_error) and self._testing:
				return False

			# finally, check hashes_not_found with hashes not found locally - if there is an intersection, we stop the upload
			# because if a hash is not found and is not on the server, there is no data to upload
			all_hashes = map(lambda iov : iov["payload_hash"], self.data_to_send["iovs"])
			hashes_not_found = check_hashes_response["hashes_not_found"]
			hashes_found = list(set(all_hashes) - set(hashes_not_found))
			self._outputter.write("Checking for IOVs that have no Payload locally or on the server.", output.VERBOSE)
			# check if any hashes not found on the server is used in the local SQLite database
			for hash_not_found in hashes_not_found:
				if hash_not_found in self.hashes_with_no_local_payload:
					return self.exit_upload("IOV with hash '%s' does not have a Payload locally or on the server.  Cannot continue." % hash_not_found)

			for hash_found in hashes_found:
				if hash_found in self.hashes_with_no_local_payload:
					self._outputter.write("Payload with hash %s on server, so can upload IOV." % hash_found, output.VERBOSE)
			
			self._outputter.write("Found %i Payloads in remote server" % len(hashes_found), output.INFO)
			self._outputter.write("Found %i Payloads not in remote server" % len(hashes_not_found), output.INFO)

			self._outputter.write("All IOVs either come with Payloads or point to a Payload already on the server.", output.VERBOSE)

		except errors.NoMoreRetriesException as no_more_retries:
			# for now, just write the log if we get a NoMoreRetriesException
			return self.exit_upload("Ran out of retries trying to check hashes of payloads to send, where the limit was 3.")
		except Exception as e:
			# something went wrong that we have no specific exception for, so just exit and output the traceback if --debug is set.
			self._outputter.write(traceback.format_exc(), output.ERROR)

			if not(self._verbose):
				self._outputter.write("Something went wrong that isn't handled by code - to get the traceback, run again with --verbose.")
			else:
				self._outputter.write("Something went wrong that isn't handled by code - the traceback is above.")

			return self.exit_upload()

		"""
		Send the payloads the server told us about in the previous step (returned from get_hashes_to_send)
		exception handling is done inside this method, since it calls a method itself for each payload.
		"""
		send_payloads_response = self.send_payloads(check_hashes_response["hashes_not_found"], self.upload_session_id)
		if self._testing and not(send_payloads_response):
			return False

		"""
		Final stage - send metadata to server (since the payloads are there now)
		if this is successful, once it finished the upload session is closed on the server and the tag lock is released.
		"""
		try:

			# note that the response (in send_metadata_response) is already decoded from base64 by the response check decorator
			send_metadata_response = self.send_metadata(self.upload_session_id)
			
			no_error = self.check_response_for_error_key(send_metadata_response)
			if not(no_error) and self._testing:
				return False

			try:
				self._outputter.write(send_metadata_response["summary"], output.INFO)
			except KeyError:
				pass

			# we have to call this explicitly here since check_response_for_error_key only writes the log file
			# if an error has occurred, whereas it should always be written here
			self.write_server_side_log(self._log_data)

		except errors.NoMoreRetriesException as no_more_retries:
			return self.exit_upload("Ran out of retries trying to send metadata, where the limit was 3.")
		except Exception as e:
			# something went wrong that we have no specific exception for, so just exit and output the traceback if --debug is set.
			self._outputter.write(traceback.format_exc(), output.ERROR)

			if not(self._verbose):
				self._outputter.write("Something went wrong that isn't handled by code - to get the traceback, run again with --verbose.")
			else:
				self._outputter.write("Something went wrong that isn't handled by code - the traceback is above.")

			return self.exit_upload()

		# close client side log handle
		self._handle.close()

		# if we're running the testing script, return True to say the upload has worked
		if self._testing:
			return True

	@check_response(check="json")
	def get_upload_session_id(self):
		"""
		Open an upload session on the server, and get a unique token back that we can use to authenticate for all future requests,
		as long as the upload session is still open.
		"""
		self._outputter.write("Getting upload session.",  output.VERBOSE)

		# send password in the body so it can be encrypted over https
		# username and password are taken from the netrc file
		# at this point, the value in username_or_token is always a username, since
		# this method's end result is obtaining a token.
		body_data = base64.b64encode(json.dumps(
				{
					"destinationTag" : self.data_to_send["destinationTags"].keys()[0],
					"username_or_token" : self.data_to_send["username"],
					"password" : self.data_to_send["password"]
				}
			))

		url_data = {"database" : self.data_to_send["destinationDatabase"]}

		query = url_query(url=self._SERVICE_URL + "get_upload_session/", body=body_data, url_data=url_data)
		response = query.send()
		return response

	@check_response(check="json")
	def close_upload_session(self, upload_session_id):
		"""
		Close an upload session on the server by calling its close_upload_session end-point.
		This is done if there is an error on the client-side.
		"""
		self._outputter.write("An error occurred - closing the upload session on the server.")
		url_data = {"database" : self.data_to_send["destinationDatabase"], "upload_session_id" : upload_session_id}
		query = url_query(url=self._SERVICE_URL + "close_upload_session/", url_data=url_data)
		response = query.send()
		return response

	@check_response(check="json")
	def get_fcsr_from_server(self, upload_session_id):
		"""
		Execute the HTTPs request to ask the server for the FCSR.

		Note: we do this in a separate function we so we can do the decoding check for json data with check_response.
		"""
		# tiny amount of client-side logic here - all of the work is done on the server
		url_data = {
						"database" : self.data_to_send["destinationDatabase"],
						"upload_session_id" : upload_session_id,
						"destinationTag" : self.data_to_send["destinationTags"].keys()[0],
						"sourceTagSync" : self.data_to_send["fcsr_filter"]
					}
		query = url_query(url=self._SERVICE_URL + "get_fcsr/", url_data=url_data)
		result = query.send()
		return result

	def filter_iovs_by_fcsr(self, upload_session_id):
		"""
		Ask for the server for the FCSR based on the synchronization type of the source Tag.
		Then, modify the IOVs (possibly remove some) based on the FCSR we received.
		This is useful in the case that most IOVs have different payloads, and our FCSR is close to the end of the range the IOVs cover.
		"""
		self._outputter.write("Getting the First Condition Safe Run for the current sync type.")

		fcsr_data = self.get_fcsr_from_server(upload_session_id)
		fcsr = fcsr_data["fcsr"]
		fcsr_changed = fcsr_data["fcsr_changed"]
		new_sync = fcsr_data["new_sync"]

		if fcsr_changed:
			self._outputter.write("Synchronization '%s' given was changed to '%s' to match destination Tag." % (self.data_to_send["fcsr_filter"], new_sync))

		self._outputter.write("Synchronization '%s' gave FCSR %d for FCSR Filtering."\
							% (self.data_to_send["fcsr_filter"], friendly_since(self.data_to_send["inputTagData"]["time_type"], fcsr)))

		"""
		There may be cases where this assumption is not correct (that we can reassign since if fcsr > since)
		Only set since to fcsr from server if the fcsr is further along than the user is trying to upload to
		Note: this applies to run, lumi and timestamp run_types.
		"""

		# if the fcsr is above the since given by the user, we need to set the user since to the fcsr
		if fcsr > self.data_to_send["since"]:
			# check if we're uploading to offline sync - if so, then user since must be >= fcsr, so we should report an error
			if self.data_to_send["fcsr_filter"].lower() == "offline":
				self._outputter.write("If you're uploading to offline, you can't upload to a since < FCSR.\nNo upload has been processed.")
				self.exit_upload()
			self.data_to_send["since"] = fcsr

		self._outputter.write("Final FCSR after comparison with FCSR received from server is %d."\
								% friendly_since(self.data_to_send["inputTagData"]["time_type"], int(self.data_to_send["since"])))

		"""
		Post validation processing assuming destination since is now valid.

		Because we don't have an sqlite database to query (everything's in a dictionary),
		we have to go through the IOVs manually find the greatest since that's less than
		the destination since.

		Purpose of this algorithm: move any IOV sinces that we can use up to the fcsr without leaving a hole in the Conditions coverage
		"""
		
		max_since_below_dest = self.data_to_send["iovs"][0]["since"]
		for (i, iov) in enumerate(self.data_to_send["iovs"]):
			if self.data_to_send["iovs"][i]["since"] <= self.data_to_send["since"] and self.data_to_send["iovs"][i]["since"] > max_since_below_dest:
				max_since_below_dest = self.data_to_send["iovs"][i]["since"]

		# only select iovs that have sinces >= max_since_below_dest
		# and then shift any IOVs left to the destination since
		self.data_to_send["iovs"] = [iov for iov in self.data_to_send["iovs"] if iov["since"] >= max_since_below_dest]
		for (i, iov) in enumerate(self.data_to_send["iovs"]):
			if self.data_to_send["iovs"][i]["since"] < self.data_to_send["since"]:
				self.data_to_send["iovs"][i]["since"] = self.data_to_send["since"]

		# modify insertion_time of iovs
		new_time = to_timestamp(datetime.now())
		for (i, iov) in enumerate(self.data_to_send["iovs"]):
			self.data_to_send["iovs"][i]["insertion_time"] = new_time

	def get_all_hashes(self):
		"""
		Get all the hashes from the dictionary of IOVs we have from the SQLite file.
		"""
		self._outputter.write("\tGetting list of all hashes found in SQLite database.", output.DEBUG)
		hashes = map(lambda iov : iov["payload_hash"], self.data_to_send["iovs"])
		self._outputter.write("Found %i local Payload(s) referenced in IOVs"%len(hashes), output.INFO)
		return hashes

	@check_response(check="json")
	def get_hashes_to_send(self, upload_session_id):
		"""
		Get the hashes of the payloads we want to send that the server doesn't have yet.
		"""
		self._outputter.write("Getting list of hashes that the server does not have Payloads for, to send to server.", output.DEBUG)
		post_data = json.dumps(self.get_all_hashes())
		url_data = {"database" : self.data_to_send["destinationDatabase"], "upload_session_id" : upload_session_id}
		query = url_query(url=self._SERVICE_URL + "check_hashes/", url_data=url_data, body=post_data)
		response = query.send()
		return response

	def send_payloads(self, hashes, upload_session_id):
		"""
		Send a list of payloads corresponding to hashes we got from the SQLite file and filtered by asking the server.
		"""
		# if we have no hashes, we can't send anything
		# but don't exit since it might mean all the Payloads were already on the server
		if len(hashes) == 0:
			self._outputter.write("No payloads to send - moving to IOV upload.")
			return True
		else:
			self._outputter.write("Sending payloads of hashes not found:")
			# construct connection string for local SQLite database file
			database = ("sqlite://%s" % os.path.abspath(self.sqlite_file_name)) if isinstance(self.sqlite_file_name, str) else self.sqlite_file_name
			# create CondDBFW connection that maps blobs - as we need to query for payload BLOBs (disabled by default in CondDBFW)
			self._outputter.write("\tConnecting to input SQLite database.")
			con = querying.connect(database, map_blobs=True)

			# query for the Payloads
			self._outputter.write("\tGetting Payloads from SQLite database based on list of hashes.")
			payloads = con.payload(hash=hashes)
			# if we get a single Payload back, put it in a list and turn it into a json_list
			if payloads.__class__ != data_sources.json_list:
				payloads = data_sources.json_data_node.make([payloads])

			# close the session with the SQLite database file - we won't use it again
			con.close_session()

			# if found some Payloads, send them
			if payloads:
				# Note: there is an edge case in which the SQLite file could have been queried
				# to delete the Payloads since we queried it for IOV hashes.  This may be handled in the next iteration.
				# send http post with data blob in body, and everything else as URL parameters
				# convert Payload to a dictionary - we can put most of this into the URL of the HTTPs request
				dicts = payloads.as_dicts()
				self._outputter.write("Uploading Payload BLOBs:")

				# for each payload, send the BLOB to the server
				for n, payload in enumerate(dicts):
					self._outputter.write("\t(%d/%d) Sending payload with hash '%s'." % (n+1, len(dicts), payload["hash"]))
					response = self.send_blob(payload, upload_session_id)
					# check response for errors
					no_error = self.check_response_for_error_key(response, exit_if_error=True)
					if not(no_error):
						return False
					self._outputter.write("\tPayload sent - moving to next one.")
				self._outputter.write("All Payloads uploaded.")
				return True
			else:
				return False

	@check_response(check="json")
	def send_blob(self, payload, upload_session_id):
		"""
		Send the BLOB of a payload over HTTP.
		The BLOB is put in the request body, so no additional processing has to be done on the server side, apart from decoding from base64.
		"""
		# encode the BLOB data of the Payload to make sure we don't send a character that will influence the HTTPs request
		blob_data = base64.b64encode(payload["data"])

		url_data = {"database" : self.data_to_send["destinationDatabase"], "upload_session_id" : upload_session_id}

		# construct the data to send in the body and header of the HTTPs request
		for key in payload.keys():
			# skip blob
			if key != "data":
				if key == "insertion_time":
					url_data[key] = to_timestamp(payload[key])
				else:
					url_data[key] = payload[key]

		request = url_query(url=self._SERVICE_URL + "store_payload/", url_data=url_data, body=blob_data)

		# send the request and return the response
		# Note - the url_query module will handle retries, and will throw a NoMoreRetriesException if it runs out
		try:
			request_response = request.send()
			return request_response
		except Exception as e:
			# make sure we don't try again - if a NoMoreRetriesException has been thrown, retries have run out
			if isinstance(e, errors.NoMoreRetriesException):
				self._outputter.write("\t\t\tPayload with hash '%s' was not uploaded because the maximum number of retries was exceeded." % payload["hash"])
				self._outputter.write("Payload with hash '%s' was not uploaded because the maximum number of retries was exceeded." % payload["hash"])
			return json.dumps({"error" : str(e), "traceback" : traceback.format_exc()})

	@check_response(check="json")
	def send_metadata(self, upload_session_id):
		"""
		Final part of the upload process - send the Conditions metadata (Tag, IOVs - not upload metadata).
		The server closes the session (and releases the tag lock) after processing has been completed.
		"""

		# set user text if it's empty
		if self.data_to_send["userText"] in ["", None]:
			self.data_to_send["userText"] = "Tag '%s' uploaded from CondDBFW client." % self.data_to_send["destinationTags"].keys()[0]

		self._outputter.write("Sending metadata to server - see server_side_log at server_side_logs/upload_log_%s for details on metadata processing on server side."\
							% self.upload_session_id, output.VERBOSE)

		# sent the HTTPs request to the server
		url_data = {"database" : self.data_to_send["destinationDatabase"], "upload_session_id" : upload_session_id}
		request = url_query(url=self._SERVICE_URL + "upload_metadata/", url_data=url_data, body=json.dumps(self.data_to_send))
		response = request.send()
		self._outputter.write("Response received - conditions upload process complete.", output.VERBOSE)
		return response

if __name__ == "__main__":
	"""
	This code should only be executed for testing.
	"""
	import sys
	from .uploadConditions import parse_arguments

	print(
"""
This code should only be executed for testing.
Any uploads done by the user should be done by calling the uploadConditions.py script.
See https://cms-conddb-dev.cern.ch/cmsDbCondUpload for information on how to obtain the correct version.
"""
	)

	upload_metadata = parse_arguments()

	upload_metadata["sqlite_file"] = upload_metadata.get("sourceDB")

	# make new dictionary, and copy over everything except "metadata_source"
	upload_metadata_argument = {}
	for (key, value) in upload_metadata.items():
		if key != "metadata_source":
			upload_metadata_argument[key] = value

	upload_metadata["metadata_source"] = data_sources.json_data_node.make(upload_metadata_argument)

	upload_controller = uploader(**upload_metadata)

	result = upload_controller.upload()