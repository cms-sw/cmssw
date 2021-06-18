#!/usr/bin/env python
"""
Primary Author:
Joshua Dawes - CERN, CMS - The University of Manchester

Debugging, Integration and Maintenance:
Andres Cardenas - CERN, CMS - Universidad San Francisco

Upload script wrapper - controls the automatic update system.

Note: the name of the file follows a different convention to the others because it should be the same as the current upload script name.

Takes user arguments and passes them to the main upload module CondDBFW.uploads, once the correct version exists.

1. Ask the server corresponding to the database we're uploading to which version of CondDBFW it has (query the /conddbfw_version/ url).
2. Decide which directory that we can write to - either the current local directory, or /tmp/random_string/.
3. Pull the commit returned from the server into the directory from step 2.
4. Invoke the CondDBFW.uploads module with the arguments given to this script.

"""

__version__ = 1

try: 
	from CondCore.Utilities.CondDBFW.url_query import url_query
except:
	print("ERROR: Could not access the url query utiliy. Yoy are probably not in a CMSSW environment.")
	exit(-1)
try:
	from StringIO import StringIO
except:
	pass
import traceback
import sys
import os
import json
import subprocess
import optparse
import netrc
import shutil
import getpass
import errno
import sqlite3


horizontal_rule = "="*60

def run_upload(**parameters):
	"""
	Imports CondDBFW.uploads and runs the upload with the upload metadata obtained.
	"""
	try:
		import CondCore.Utilities.CondDBFW.uploads as uploads
	except Exception as e:
		traceback.print_exc()
		exit("CondDBFW or one of its dependencies could not be imported.\n"\
			+ "If the CondDBFW directory exists, you are likely not in a CMSSW environment.")
	# we have CondDBFW, so just call the module with the parameters given in the command line
	uploader = uploads.uploader(**parameters)
	result = uploader.upload()

def getInput(default, prompt = ''):
    '''Like raw_input() but with a default and automatic strip().
    '''

    answer = raw_input(prompt)
    if answer:
        return answer.strip()

    return default.strip()


def getInputWorkflow(prompt = ''):
    '''Like getInput() but tailored to get target workflows (synchronization options).
    '''

    while True:
        workflow = getInput(defaultWorkflow, prompt)

        if workflow in frozenset(['offline', 'hlt', 'express', 'prompt', 'pcl']):
            return workflow

        print('Please specify one of the allowed workflows. See above for the explanation on each of them.')


def getInputChoose(optionsList, default, prompt = ''):
    '''Makes the user choose from a list of options.
    '''

    while True:
        index = getInput(default, prompt)

        try:
            return optionsList[int(index)]
        except ValueError:
            print('Please specify an index of the list (i.e. integer).')
        except IndexError:
            print('The index you provided is not in the given list.')


def getInputRepeat(prompt = ''):
    '''Like raw_input() but repeats if nothing is provided and automatic strip().
    '''

    while True:
        answer = raw_input(prompt)
        if answer:
            return answer.strip()

        print('You need to provide a value.')

def runWizard(basename, dataFilename, metadataFilename):
    while True:
        print('''\nWizard for metadata for %s

I will ask you some questions to fill the metadata file. For some of the questions there are defaults between square brackets (i.e. []), leave empty (i.e. hit Enter) to use them.''' % basename)

        # Try to get the available inputTags
        try:
            dataConnection = sqlite3.connect(dataFilename)
            dataCursor = dataConnection.cursor()
            dataCursor.execute('select name from sqlite_master where type == "table"')
            tables = set(zip(*dataCursor.fetchall())[0])

            # only conddb V2 supported...
            if 'TAG' in tables:
                dataCursor.execute('select NAME from TAG')
            # In any other case, do not try to get the inputTags
            else:
                raise Exception()

            inputTags = dataCursor.fetchall()
            if len(inputTags) == 0:
                raise Exception()
            inputTags = list(zip(*inputTags))[0]

        except Exception:
            inputTags = []

        if len(inputTags) == 0:
            print('\nI could not find any input tag in your data file, but you can still specify one manually.')

            inputTag = getInputRepeat(
                '\nWhich is the input tag (i.e. the tag to be read from the SQLite data file)?\ne.g. BeamSpotObject_ByRun\ninputTag: ')

        else:
            print('\nI found the following input tags in your SQLite data file:')
            for (index, inputTag) in enumerate(inputTags):
                print('   %s) %s' % (index, inputTag))

            inputTag = getInputChoose(inputTags, '0',
                                      '\nWhich is the input tag (i.e. the tag to be read from the SQLite data file)?\ne.g. 0 (you select the first in the list)\ninputTag [0]: ')

        databases = {
            'oraprod': 'oracle://cms_orcon_prod/CMS_CONDITIONS',
			'prod': 'oracle://cms_orcon_prod/CMS_CONDITIONS',
            'oradev': 'oracle://cms_orcoff_prep/CMS_CONDITIONS',
			'prep': 'oracle://cms_orcoff_prep/CMS_CONDITIONS',
        }

        destinationDatabase = ''
        ntry = 0
        print('\nWhich is the destination database where the tags should be exported?')
        print('\n%s) %s' % ('oraprod', databases['oraprod']))
        print('\n%s) %s' % ('oradev', databases['oradev']))
            
        while ( destinationDatabase not in databases.values() ): 
            if ntry==0:
                inputMessage = \
                '\nPossible choices: oraprod or oradev \ndestinationDatabase: '
            elif ntry==1:
                inputMessage = \
                '\nPlease choose one of the two valid destinations: oraprod or oradev \ndestinationDatabase: '
            else:
                raise Exception('No valid destination chosen. Bailing out...')
			
            databaseInput = getInputRepeat(inputMessage).lower()
            if databaseInput in databases.keys():
                destinationDatabase = databases[databaseInput]
            ntry += 1

        while True:
            since = getInput('',
                             '\nWhich is the given since? (if not specified, the one from the SQLite data file will be taken -- note that even if specified, still this may not be the final since, depending on the synchronization options you select later: if the synchronization target is not offline, and the since you give is smaller than the next possible one (i.e. you give a run number earlier than the one which will be started/processed next in prompt/hlt/express), the DropBox will move the since ahead to go to the first safe run instead of the value you gave)\ne.g. 1234\nsince []: ')
            if not since:
                since = None
                break
            else:
                try:
                    since = int(since)
                    break
                except ValueError:
                    print('The since value has to be an integer or empty (null).')

        userText = getInput('',
                            '\nWrite any comments/text you may want to describe your request\ne.g. Muon alignment scenario for...\nuserText []: ')

        destinationTags = {}
        while True:
            destinationTag = getInput('',
                                      '\nWhich is the next destination tag to be added (leave empty to stop)?\ne.g. BeamSpotObjects_PCL_byRun_v0_offline\ndestinationTag []: ')
            if not destinationTag:
                if len(destinationTags) == 0:
                    print('There must be at least one destination tag.')
                    continue
                break

            if destinationTag in destinationTags:
                print(
                    'You already added this destination tag. Overwriting the previous one with this new one.')

            destinationTags[destinationTag] = {
            }

        metadata = {
            'destinationDatabase': destinationDatabase,
            'destinationTags': destinationTags,
            'inputTag': inputTag,
            'since': since,
            'userText': userText,
        }

        metadata = json.dumps(metadata, sort_keys=True, indent=4)
        print('\nThis is the generated metadata:\n%s' % metadata)

        if getInput('n',
                    '\nIs it fine (i.e. save in %s and *upload* the conditions if this is the latest file)?\nAnswer [n]: ' % metadataFilename).lower() == 'y':
            break
    print('Saving generated metadata in %s...'% metadataFilename)
    with open(metadataFilename, 'wb') as metadataFile:
        metadataFile.write(metadata)

def parse_arguments():
	# read in command line arguments, and build metadata dictionary from them
	parser = optparse.OptionParser(description="CMS Conditions Upload Script in CondDBFW.",
		usage = 'Usage: %prog [options] <file>')

	# metadata arguments
	parser.add_option("-i", "--inputTag", type=str,\
						help="Tag to take IOVs + Payloads from in --sourceDB.")
	parser.add_option("-t", "--destinationTag", type=str,\
						help="Tag to copy IOVs + Payloads to in --destDB.")
	parser.add_option("-D", "--destinationDatabase", type=str,\
						help="Database to copy IOVs + Payloads to.")
	parser.add_option("-s", "--since", type=int,\
						help="Since to take IOVs from.")
	parser.add_option("-u", "--userText", type=str,\
						help="Description of --destTag (can be empty).")

	# non-metadata arguments
	parser.add_option("-m", "--metadataFile", type=str, help="Metadata file to take metadata from.")

	parser.add_option("-d", "--debug", action="store_true", default=False)
	parser.add_option("-v", "--verbose", action="store_true", default=False)
	parser.add_option("-T", "--testing", action="store_true")
	parser.add_option("--fcsr-filter", type=str, help="Synchronization to take FCSR from for local filtering of IOVs.")

	parser.add_option("-n", "--netrc", help = 'The netrc host (machine) from where the username and password will be read.')
	
	parser.add_option("-a", "--authPath", help = 'The path of the .netrc file for the authentication. Default: $HOME')

	parser.add_option("-H", "--hashToUse")

	parser.add_option("-S", "--server")

	parser.add_option("-o", "--review-options", action="store_true")

	parser.add_option("-r", "--replay-file")

	(command_line_data, arguments) = parser.parse_args()

	if len(arguments) < 1:
		if command_line_data.hashToUse == None:
			parser.print_help()
			exit(-2)
	
	command_line_data.sourceDB = arguments[0]

	if command_line_data.replay_file:
		dictionary = json.loads("".join(open(command_line_data.replay_file, "r").readlines()))
		command_line_data.tier0_response = dictionary["tier0_response"]

	# default is the production server, which can point to either database anyway
	server_alias_to_url = {
		"prep" : "https://cms-conddb-dev.cern.ch/cmsDbCondUpload/",
		"dev" : "https://cms-conddb-dev.cern.ch/cmsDbCondUpload/",
		"prod" : "https://cms-conddb.cern.ch/cmsDbCondUpload/"
	}

	# if prep, prod or None were given, convert to URLs in dictionary server_alias_to_url
	# if not, assume a URL has been given and use this instead
	if command_line_data.server in server_alias_to_url.keys():
		command_line_data.server = server_alias_to_url[command_line_data.server]

	# resolve destination databases
	database_alias_to_connection = {
		"prep": "oracle://cms_orcoff_prep/CMS_CONDITIONS",
		"dev": "oracle://cms_orcoff_prep/CMS_CONDITIONS",
		"prod": "oracle://cms_orcon_adg/CMS_CONDITIONS"
	}
	
	if command_line_data.destinationDatabase in database_alias_to_connection.keys():
		command_line_data.destinationDatabase = database_alias_to_connection[command_line_data.destinationDatabase]


	# use netrc to get username and password
	try:
		netrc_file = command_line_data.netrc
		auth_path = command_line_data.authPath
		if not auth_path is None:
			if netrc_file is None:
				netrc_file = os.path.join(auth_path,'.netrc')
			else:
				netrc_file = os.path.join(auth_path, netrc_file)

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

	# Hash to use, entirely from command line
	if command_line_data.hashToUse != None:
		command_line_data.userText = ""
		metadata_dictionary = command_line_data.__dict__
	elif command_line_data.metadataFile == None:
		if command_line_data.sourceDB != None and (command_line_data.inputTag == None or command_line_data.destinationTag == None or command_line_data.destinationDatabase == None):
			basepath = command_line_data.sourceDB.rsplit('.db', 1)[0].rsplit('.txt', 1)[0]
			basename = os.path.basename(basepath)
			dataFilename = '%s.db' % basepath
			metadataFilename = '%s.txt' % basepath

			# Data file
			try:
				with open(dataFilename, 'rb') as dataFile:
					pass
			except IOError as e:
				errMsg = 'Impossible to open SQLite data file %s' %dataFilename
				print( errMsg )
				ret['status'] = -3
				ret['error'] = errMsg
				return ret

			# Metadata file

			command_line_data.sourceDB = dataFilename

			try:
				with open(metadataFilename, 'rb') as metadataFile:
					pass
			except IOError as e:
				if e.errno != errno.ENOENT:
					errMsg = 'Impossible to open file %s (for other reason than not existing)' %metadataFilename
					ret = {}
					ret['status'] = -4
					ret['error'] = errMsg
					exit (ret)

				if getInput('y', '\nIt looks like the metadata file %s does not exist and not enough parameters were received in the command line. Do you want me to create it and help you fill it?\nAnswer [y]: ' % metadataFilename).lower() != 'y':
					errMsg = 'Metadata file %s does not exist' %metadataFilename
					ret = {}
					ret['status'] = -5
					ret['error'] = errMsg
					exit(ret)
				# Wizard
				runWizard(basename, dataFilename, metadataFilename)
			command_line_data.metadataFile = metadataFilename
		else:
			command_line_data.userText = command_line_data.userText\
										if command_line_data.userText != None\
										else str(raw_input("Tag's description [can be empty]:"))
			metadata_dictionary = command_line_data.__dict__

	if command_line_data.metadataFile != None:
		metadata_dictionary = json.loads("".join(open(os.path.abspath(command_line_data.metadataFile), "r").readlines()))
		metadata_dictionary["username"] = username
		metadata_dictionary["password"] = password
		metadata_dictionary["userText"] = metadata_dictionary.get("userText")\
											if metadata_dictionary.get("userText") != None\
											else str(raw_input("Tag's description [can be empty]:"))

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

	if metadata_dictionary["server"] == None:
		if metadata_dictionary["destinationDatabase"] == "oracle://cms_orcoff_prep/CMS_CONDITIONS":
			metadata_dictionary["server"] = server_alias_to_url["prep"]
		else:
			metadata_dictionary["server"] = server_alias_to_url["prod"]

	return metadata_dictionary

def get_version(url):
	query = url_query(url=url + "script_version/")
	response = query.send()
	return response


if __name__ == "__main__":

	upload_metadata = parse_arguments()

	# upload_metadata should be used to decide the service url
	final_service_url = upload_metadata["server"]
	try:
		response = get_version(final_service_url)
		server_version = json.loads(response)
	except Exception as e:
		print(horizontal_rule)
		print(e)
		print("Could not connect to server at %s"%final_service_url)
		print("If you specified a server please check it is correct. If that is not the issue please contact the AlcaDB team.")
		print(horizontal_rule)
		exit(1)

	if server_version["version"] != __version__:
		print(horizontal_rule)
		print("Local upload script is different than server version. Please run the following command to get the latest script.")
		print("curl --insecure -o uploadConditions.py %sget_upload_script/ && chmod +x uploadConditions.py;"%final_service_url)
		print(horizontal_rule)
		exit(1)

	import CondCore.Utilities.CondDBFW.data_sources as data_sources

	upload_metadata["sqlite_file"] = upload_metadata.get("sourceDB")

	try:
		os.mkdir('upload_logs')
	except OSError as e:
		pass

	# make new dictionary, and copy over everything except "metadata_source"
	upload_metadata_argument = {}
	for (key, value) in upload_metadata.items():
		if key != "metadata_source":
			upload_metadata_argument[key] = value

	upload_metadata["metadata_source"] = data_sources.json_data_node.make(upload_metadata_argument)
	try:
		# pass dictionary as arguments to match keywords - the constructor has a **kwargs parameter to deal with stray arguments
		run_upload(**upload_metadata)
		print(horizontal_rule)
		print("Process completed without issues. Please check logs for further details.")
		print(horizontal_rule)
	except SystemExit as e:
		print(horizontal_rule)
		print("Process exited abnormally. Please check logs for details.")
		print(horizontal_rule)
		exit(1)
	exit(0)
