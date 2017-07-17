#!/usr/bin/env python 
"""

Joshua Dawes - CERN, CMS - The University of Manchester

Upload script wrapper - controls the automatic update system.

Note: the name of the file follows a different convention to the others because it should be the same as the current upload script name.

Takes user arguments and passes them to the main upload module CondDBFW.uploads, once the correct version exists.

1. Ask the server corresponding to the database we're uploading to which version of CondDBFW it has (query the /conddbfw_version/ url).
2. Decide which directory that we can write to - either the current local directory, or /tmp/random_string/.
3. Pull the commit returned from the server into the directory from step 2.
4. Invoke the CondDBFW.uploads module with the arguments given to this script.

"""

import pycurl
from StringIO import StringIO
import traceback
import sys
import os
import json
import subprocess
import argparse
import netrc
import shutil
import getpass

def get_version_info(url):
	"""
	Queries the server-side for the commit hash it is currently using.
	Note: this is the commit hash used by /data/services/common/CondDBFW on the server-side.
	"""
	request = pycurl.Curl()
	request.setopt(request.CONNECTTIMEOUT, 60)
	user_agent = "User-Agent: ConditionWebServices/1.0 python/%d.%d.%d PycURL/%s" % (sys.version_info[ :3 ] + (pycurl.version_info()[1],))
	request.setopt(request.USERAGENT, user_agent)
	# we don't need to verify who signed the certificate or who the host is
	request.setopt(request.SSL_VERIFYPEER, 0)
	request.setopt(request.SSL_VERIFYHOST, 0)
	response_buffer = StringIO()
	request.setopt(request.WRITEFUNCTION, response_buffer.write)
	request.setopt(request.URL, url + "conddbfw_version/")
	request.perform()
	return json.loads(response_buffer.getvalue())

def get_local_commit_hash():
	"""
	Gets the commit hash used by the local repository CondDBFW/.git/.
	"""
	directory = os.path.abspath("CondDBFW")

	# get the commit hash of the code in `directory`
	# by reading the .commit_hash file
	try:
		commit_hash_file_handle = open(os.path.join(directory, ".commit_hash"), "r")
		commit_hash = commit_hash_file_handle.read().strip()

		# validate length of the commit hash
		if len(commit_hash) != 40:
			print("Commit hash found is not valid.  Must be 40 characters long.")
			exit()

		#commit_hash = run_in_shell("git --git-dir=%s rev-parse HEAD" % (os.path.join(directory, ".git")), shell=True).strip()

		return commit_hash
	except Exception:
		return None

def get_directory_to_pull_to(default_directory, commit_hash):
	"""
	Finds out which directory we can safely use - either CondDBFW/ or a temporary directory.
	"""
	# try to write a file (and then delete it)
	try:
		handle = open(os.path.join(default_directory, "test_file"), "w")
		handle.write("test")
		handle.close()
		os.remove(os.path.join(default_directory, "test_file"))
		sys.path.insert(0, default_directory)
		return default_directory
	except IOError as io:
		# cannot write to default directory, so set up a directory in /tmp/
		new_path = os.path.join("tmp", commit_hash[0:10])
		if not(os.path.exists(new_path)):
			os.mkdir(new_path)
			sys.path.insert(0, new_path)
			return new_path
		else:
			# for now, fail
			exit("Can't find anywhere to pull the new code base to.")

horizontal_rule = "="*60

def pull_code_from_git(target_directory, repository_url, hash):
	"""
	Pulls CondDBFW from the git repository specified by the upload server.
	"""
	# make directory
	target = os.path.abspath(target_directory)
	sys.path.append(target)
	conddbfw_directory = os.path.join(target, "CondDBFW")
	git_directory = os.path.join(conddbfw_directory, ".git")
	if not(os.path.exists(conddbfw_directory)):
		os.mkdir(conddbfw_directory)
	else:
		# if the directory exists, it may contain things - prompt the user
		force_pull = str(raw_input("CondDBFW directory isn't empty - empty it, and update to new version? [y/n] "))
		if force_pull == "y":
			# empty directory and delete it
			run_in_shell("rm -rf CondDBFW", shell=True)
			# remake the directory - it will be empty
			os.mkdir(conddbfw_directory)

	print("Pulling code back from repository...")
	print(horizontal_rule)

	run_in_shell("git --git-dir=%s clone %s CondDBFW" % (git_directory, repository_url), shell=True)
	# --force makes sure we ignore any conflicts that
	# could occur and overwrite everything in the checkout
	run_in_shell("cd %s && git checkout --force -b version_used %s" % (conddbfw_directory, hash), shell=True)

	# write the hash to a file in the CondDBFW directory so we can delete the git repository
	hash_file_handle = open(os.path.join(conddbfw_directory, ".commit_hash"), "w")
	hash_file_handle.write(hash)
	hash_file_handle.close()

	# can now delete .git directory
	shutil.rmtree(git_directory)

	print(horizontal_rule)
	print("Creating local log directories (if required)...")
	if not(os.path.exists(os.path.join(target, "upload_logs"))):
		os.mkdir(os.path.join(target, "upload_logs"))
	if not(os.path.exists(os.path.join(target, "server_side_logs"))):
		os.mkdir(os.path.join(target, "server_side_logs"))
	print("Finished with log directories.")
	print("Update of CondDBFW complete.")

	print(horizontal_rule)

	return True

def run_in_shell(*popenargs, **kwargs):
	"""
	Runs string-based commands in the shell and returns the result.
	"""
	out = subprocess.PIPE if kwargs.get("stdout") == None else kwargs.get("stdout")
	new_kwargs = kwargs
	if new_kwargs.get("stdout"):
		del new_kwargs["stdout"]
	process = subprocess.Popen(*popenargs, stdout=out, **new_kwargs)
	stdout = process.communicate()[0]
	returnCode = process.returncode
	cmd = kwargs.get('args')
	if cmd is None:
		cmd = popenargs[0]
	if returnCode:
		raise subprocess.CalledProcessError(returnCode, cmd)
	return stdout

def run_upload(**parameters):
	"""
	Imports CondDBFW.uploads and runs the upload with the upload metadata obtained.
	"""
	try:
		import CondDBFW.uploads as uploads
	except Exception as e:
		traceback.print_exc()
		exit("CondDBFW or one of its dependencies could not be imported.\n"\
			+ "If the CondDBFW directory exists, you are likely not in a CMSSW environment.")
	# we have CondDBFW, so just call the module with the parameters given in the command line
	uploader = uploads.uploader(**parameters)
	result = uploader.upload()

def parse_arguments():
	# read in command line arguments, and build metadata dictionary from them
	parser = argparse.ArgumentParser(prog="cmsDbUpload client", description="CMS Conditions Upload Script in CondDBFW.")

	parser.add_argument("--sourceDB", type=str, help="DB to find Tags, IOVs + Payloads in.", required=False)

	# metadata arguments
	parser.add_argument("--inputTag", type=str,\
						help="Tag to take IOVs + Payloads from in --sourceDB.", required=False)
	parser.add_argument("--destinationTag", type=str,\
						help="Tag to copy IOVs + Payloads to in --destDB.", required=False)
	parser.add_argument("--destinationDatabase", type=str,\
						help="Database to copy IOVs + Payloads to.", required=False)
	parser.add_argument("--since", type=int,\
						help="Since to take IOVs from.", required=False)
	parser.add_argument("--userText", type=str,\
						help="Description of --destTag (can be empty).")

	# non-metadata arguments
	parser.add_argument("--metadataFile", "-m", type=str, help="Metadata file to take metadata from.", required=False)

	parser.add_argument("--debug", required=False, action="store_true")
	parser.add_argument("--verbose", required=False, action="store_true")
	parser.add_argument("--testing", required=False, action="store_true")
	parser.add_argument("--fcsr-filter", type=str, help="Synchronization to take FCSR from for local filtering of IOVs.", required=False)

	parser.add_argument("--netrc", required=False)

	parser.add_argument("--hashToUse", required=False)

	parser.add_argument("--server", required=False)

	parser.add_argument("--review-options", required=False, action="store_true")

	command_line_data = parser.parse_args()

	# default is the production server, which can point to either database anyway
	server_alias_to_url = {
		"prep" : "https://cms-conddb-dev.cern.ch/cmsDbCondUpload/",
		"prod" : "https://cms-conddb.cern.ch/cmsDbCondUpload/",
		None : "https://cms-conddb.cern.ch/cmsDbCondUpload/"
	}

	# if prep, prod or None were given, convert to URLs in dictionary server_alias_to_url
	# if not, assume a URL has been given and use this instead
	if command_line_data.server in server_alias_to_url.keys():
		command_line_data.server = server_alias_to_url[command_line_data.server]

	# use netrc to get username and password
	try:
		netrc_file = command_line_data.netrc
		netrc_authenticators = netrc.netrc(netrc_file).authenticators("ConditionUploader")
		if netrc_authenticators == None:
			print("Your netrc file must contain the key 'ConditionUploader'.")
			manual_input = raw_input("Do you want to try to type your credentials? ")
			if manual_input == "y":
				# ask for username and password
				username = raw_input("Username: ")
				password = getpass.getpass("Password: ")
			else:
				exit()
		else:
			print("Read your credentials from ~/.netrc.  If you want to use a different file, supply its name with the --netrc argument.")
			username = netrc_authenticators[0]
			password = netrc_authenticators[2]
	except:
		print("Couldn't obtain your credentials (either from netrc or manual input).")
		exit()

	command_line_data.username = username
	command_line_data.password = password
	# this will be used as the final destinationTags value by all input methods
	# apart from the metadata file
	command_line_data.destinationTags = {command_line_data.destinationTag:{}}

	"""
	Construct metadata_dictionary:
	Currently, this is 3 cases:

	1) An IOV is being appended to an existing Tag with an existing Payload.
	In this case, we just take all data from the command line.

	2) No metadata file is given, so we assume that ALL upload metadata is coming from the command line.

	3) A metadata file is given, hence we parse the file, and then iterate through command line arguments
	since these override the options set in the metadata file.

	"""
	if command_line_data.hashToUse != None:
		command_line_data.userText = ""
		metadata_dictionary = command_line_data.__dict__
	elif command_line_data.metadataFile == None:
		command_line_data.userText = command_line_data.userText\
									if command_line_data.userText != None\
									else str(raw_input("Tag's description [can be empty]:"))
		metadata_dictionary = command_line_data.__dict__
	else:
		metadata_dictionary = json.loads("".join(open(os.path.abspath(command_line_data.metadataFile), "r").readlines()))
		metadata_dictionary["username"] = username
		metadata_dictionary["password"] = password
		metadata_dictionary["userText"] = metadata_dictionary.get("userText")\
											if metadata_dictionary.get("userText") != None\
											else str(raw_input("Tag's description [can be empty]:"))
		# set the server to use to be the default one
		metadata_dictionary["server"] = server_alias_to_url[None]

		# go through command line options and, if they are set, overwrite entries
		for (option_name, option_value) in command_line_data.__dict__.items():
			# if the metadata_dictionary sets this, overwrite it
			if option_name != "destinationTags":
				if option_value != None or (option_value == None and not(option_name in metadata_dictionary.keys())):
					# if option_value has a value, override the metadata file entry
					# or if option_value is None but the metadata file doesn't give a value,
					# set the entry to None as well
					metadata_dictionary[option_name] = option_value
			else:
				if option_value != {None:{}}:
					metadata_dictionary["destinationTags"] = {option_value:{}}
				elif option_value == {None:{}} and not("destinationTags" in metadata_dictionary.keys()):
					metadata_dictionary["destinationTags"] = {None:{}}

	if command_line_data.review_options:
		defaults = {
			"since" : "Since of first IOV",
			"userText" : "Populated by upload process",
			"netrc" : "None given",
			"fcsr_filter" : "Don't apply",
			"hashToUse" : "Using local SQLite file instead"
		}
		print("Configuration to use for the upload:")
		for key in metadata_dictionary:
			if not(key) in ["username", "password", "destinationTag"]:
				value_to_print = metadata_dictionary[key] if metadata_dictionary[key] != None else defaults[key]
				print("\t%s : %s" % (key, value_to_print))

		if raw_input("\nDo you want to continue? [y/n] ") != "y":
			exit()

	return metadata_dictionary

if __name__ == "__main__":

	upload_metadata = parse_arguments()

	# upload_metadata should be used to decide the service url
	final_service_url = upload_metadata["server"]

	conddbfw_version = get_version_info(final_service_url)
	local_version = get_local_commit_hash()

	"""
	Todo - case where we don't have write permission in the current directory (local_version == None and hashes don't match)
	"""
	# target_directory is only used if we don't find a version of CondDBFW locally,
	# but is set here so we can access it later if we need to delete a temporary directory
	target_directory = ""
	# check if we have a persistent local version of CondDBFW
	if local_version != None:
		if conddbfw_version["hash"] == local_version:
			# no update is required, pass for now
			print("No change of version of CondDBFW is required - performing the upload.")
			# add CondDBFW to the system paths (local_version != None, so we know it's in this directory)
			sys.path.append(os.path.abspath(os.getcwd()))
		elif conddbfw_version["hash"] != local_version:
			# this is the case where CondDBFW is in the directory working_dir/CondDBFW, but there is an update available
			# CondDBFW isn't in this directory, and the local commit hash doesn't match the latest one on the server
			print("The server uses a different version of CondDBFW - changing to commit '%s' of CondDBFW." % conddbfw_version["hash"])
			shell_response = pull_code_from_git(os.getcwd(), conddbfw_version["repo"], conddbfw_version["hash"])
	else:
		# no CondDBFW version - we should pull the code and start from scratch
		# we can't look for temporary versions of it in /tmp/, since we can't guess the hash used to make the directory name
		print("No CondDBFW version found locally - pulling one.")
		target_directory = get_directory_to_pull_to(os.getcwd(), conddbfw_version["hash"])
		shell_response = pull_code_from_git(target_directory, conddbfw_version["repo"], conddbfw_version["hash"])

	import CondDBFW.data_sources as data_sources

	upload_metadata["sqlite_file"] = upload_metadata.get("sourceDB")
	
	# make new dictionary, and copy over everything except "metadata_source"
	upload_metadata_argument = {}
	for (key, value) in upload_metadata.items():
		if key != "metadata_source":
			upload_metadata_argument[key] = value

	upload_metadata["metadata_source"] = data_sources.json_data_node.make(upload_metadata_argument)

	# pass dictionary as arguments to match keywords - the constructor has a **kwargs parameter to deal with stray arguments
	run_upload(**upload_metadata)

	# if the directory was temporary, delete it
	if "tmp" in target_directory:
		print(horizontal_rule)
		print("Removing directory %s..." % target_directory)
		try:
			run_in_shell("rm -rf %s" % target_directory, shell=True)
		except Exception as e:
			print("Couldn't delete the directory %s - try to manually delete it." % target_directory)