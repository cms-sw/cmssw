#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## May 17th 2012

# This module will write the data from a passed software object
# to the database file on disk
import os
from CMGTools.Production.cmgdbApi import CmgdbApi
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/oracle/11.2.0.1.0p2/etc/profile.d/init.sh")
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/python/2.6.4-cms16/etc/profile.d/init.sh")
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/py2-cx-oracle/5.1/etc/profile.d/init.sh")

import CMGTools.Production.eostools as eostools

import sys, re
#import CMGTools.Production.cx_Oracle as cx_Oracle
import cx_Oracle

class CmgdbToolsApi(CmgdbApi):
	
	""" A class for interacting with the CMGDB database through a structured API
	"""
	
	# Log num of good files
	def addGoodFileNum(self,datasetID,number):
		"""Record the quantity of bad files
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'number' takes the number of good files as an int
		"""
		try:
			self.insertCur.execute("UPDATE "+self.schema_name+".dataset_details set number_files_good=%d where dataset_id = %d" % (number, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"
	
	# Log num of good files
	def addGroupGoodFileNum(self,group_id,number):
		"""Record the quantity of bad files
			
		'group_id' takes the unique ID of the file group
		'number' takes the number of good files as an int
		"""
		try:
			self.insertCur.execute("UPDATE "+self.schema_name+".file_group_details set number_files_good=%d where file_group_id=%d" % (number,group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"
	
	# Clear all the missing files relating to a dataset
	def clearDatasetMissingFiles(self, dsName, datasetID):
		"""Clear all missing files from CMGDB pertaining to the given dataset ID
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM 
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		Returns None
		"""
		try:
			self.insertCur.execute('DELETE FROM ' +self.schema_name+ '.missing_files WHERE dataset_id = %d' % int(datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Unable to delete missing file record"
	
	# Log num of missing files
	def addMissingFileNum(self,datasetID,number):
		"""Record the quantity of bad files
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'number' takes the number of missing files as an int
		"""
		try:
			self.insertCur.execute("UPDATE "+self.schema_name+".dataset_details set number_files_missing=%d where dataset_id = %d" % (number, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"        
	
	# Log num of good files
	def addGroupMissingFileNum(self,group_id,number):
		"""Record the quantity of bad files
			
		'group_id' takes the unique ID of the file group
		'number' takes the number of missing files as an int
		"""
		try:
			self.insertCur.execute("UPDATE "+self.schema_name+ ".file_group_details set number_files_missing=%d where file_group_id=%d" % (number,group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"
	
	# Log missing files
	def addMissingFile(self, dsName, datasetID, missingFileName, group_id):
		"""Add name of given missing file to CMGDB and link it to a given dataset with the unique CMGDB Dataset ID
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'missingFileName' takes the name of the missing file as a string
		'group_id' takes the unique ID of the file group
		Returns None
		"""
		try:
			self.insertCur.execute("INSERT INTO "+self.schema_name+ ".missing_files(dataset_id, missing_file, file_group_id) values(%d, '%s', %d)" % (datasetID, missingFileName, group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			#If exception is thrown display error message and ignore
			print 'File: '+missingFileName+ " in dataset " + dsName + " is already logged as missing on the system"
	
	# Clear bad file entries
	def clearDatasetBadFiles(self, dsName, datasetID):
		"""Clear all bad files from CMGDB pertaining to the given dataset ID
			
		'dsName' takes the name of the dataset as a string, 
		'datasetID' takes unique CMGDB Dataset ID of the dataset as an int
		Returns None
		"""
		try:
			self.insertCur.execute('DELETE FROM ' +self.schema_name+ '.bad_files WHERE dataset_id = %d' % int(datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Unable to delete file entries"
	
	# Log num of bad files
	def addBadFileNum(self,datasetID,number):
		"""Record the quantity of bad files
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'number' takes the number of bad files as an int
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set number_files_bad=%d where dataset_id = %d" % (number, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"
	
	# Log num of good files
	def addGroupBadFileNum(self,group_id,number):
		"""Record the quantity of bad files
			
		'group_id' takes the unique ID of the file group
		'number' takes the number of bad files as an int
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".file_group_details set number_files_bad=%d where file_group_id=%d" % (number,group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"
	
	# Log bad files
	def addBadFile(self, dsName, datasetID, badFileName, group_id):
		"""Add name of given bad file to CMGDB and link it to a given dataset with the unique CMGDB Dataset ID
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'badFileName' takes the name of the bad file as a string
		'group_id' takes the unique ID of the file group
		Returns None
		"""
		try:
			self.insertCur.execute("INSERT INTO "+self.schema_name+ ".bad_files(dataset_id, bad_file, file_group_id) values(%d, '%s', %d)" % (datasetID, str(badFileName),group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			#If exception is thrown display error message and ignore
			print 'File: '+badFileName+ " in dataset " + dsName + " is already logged as bad on the system"
	
	# Clear bad jobs entries
	def clearDatasetBadJobs(self, dsName, datasetID):
		"""Clear all bad jobs from CMGDB pertaining to the given dataset ID
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'datasetID' takes unique CMGDB Dataset ID of the dataset as an int
		Returns None
		"""
		try:
			self.insertCur.execute('DELETE FROM '+self.schema_name+ '.bad_jobs WHERE dataset_id = %d' % int(datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Unable to delete bad job record"
	
	# Log missing files
	def addBadJob(self, datasetID, badJobNum):
		"""Add name of given bad job to CMGDB and link it to a given dataset with the unique CMGDB Dataset ID
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'badJobNum' takes the name of the bad job as an int
		Returns None
		"""
		try:
			self.insertCur.execute('INSERT INTO '+self.schema_name+'.bad_jobs(dataset_id, bad_job) values(%d, %d)' % (datasetID, int(badJobNum)))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			#If exception is thrown display error message and ignore
			print 'Job: '+badJobName + " is already logged as bad on the system"
	
	# Log num of good files
	def addBadJobNum(self,datasetID,number):
		"""Record the quantity of bad files
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'number' takes the number of bad jobs as an int
		"""
		try:
			self.insertCur.execute("UPDATE "+self.schema_name+".dataset_details set number_jobs_bad=%d where dataset_id = %d" % (number, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"
	
	# Add a value for the number_total_jobs field
	def addTotalJobs(self, datasetID, nJobs):
		"""Record the total number of jobs
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'nJobs' takes the number of jobs as an int
		"""
		try:
			self.insertCur.execute("UPDATE "+self.schema_name+".dataset_details set number_total_jobs=%d where dataset_id = %d" % (nJobs, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			print "Dataset not found"
	
	# Log a tag against a package
	def addTag(self, package, tag):
		"""Add a tag and associated package to the 'tags' table in CMGDB with a unique ID
			
		'package' takes the name of the package as a string
		'tag' takes the name of the tag as a string
		Returns the unique CMGDB Tag ID
			
		First the function checks if the tag/package combination already exists in CMGDB
		-If yes, the tag/package unique ID is returned
		-If no, a new unique ID is created, the tag/package is added to CMGDB and the unique ID is returned
		"""
		try:
			print tag," " ,package
			tagID=self.getTagID(package, tag)
			if tagID!= None:
				return tagID
			tagID = self.selectCur.execute("select "+self.schema_name+ ".tag_id_seq.NEXTVAL from dual").fetchone()[0]
			self.insertCur.execute("INSERT INTO "+self.schema_name+".tags(tag_id, package_name, tag) values(:tagID , :package, :tag)",{"tagID":tagID,"package":package,"tag":tag})
			self.insertConn.commit()
			return tagID
		except cx_Oracle.IntegrityError:
			#If exception is thrown display error message and ignore
			print tag + " " + package + " was not added."
			return None
	
	# Adds a set to the database
	def addDataset(self, dsName,eos,lfn, fileOwner, parentID, username):
		"""Add a dataset with corresponding attributes to the 'dataset_details' table in CMGDB
			
		'dsName' takes the dataset name in CMGDB format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		'eos' takes the dataset name in EOS format as a string e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1/AODSIM/V3 
		'lfn' takes the LFN of the dataset
		'fileowner' takes the datasets owner on EOS e.g cmgtools
		'parentID' takes the unique CMGDB ID of the datasets parent. This is None if parent is a CMS dataset (like /DoubleMu/Run2011B-16Jan2012-v1/AOD)
		'username' takes the NICE username of the person publishing the dataset.
		RETURN the unique CMGDB Dataset ID
			
		First the function checks if the dataset already exists in CMGDB
		-If yes, the unique CMGDB Dataset ID is returned
		-If no, a new unique ID is created, the dataset is added to CMGDB and the unique ID is returned
		"""
		try:
			datasetID = self.getDatasetIDWithName(dsName)
			if datasetID is not None:
				print "Dataset %s is already present on the system\n" % dsName
				return datasetID
			#Insert information into database
			datasetID = self.selectCur.execute("select "+self.schema_name+ ".dataset_id_seq.NEXTVAL from dual").fetchone()[0]
			if parentID is None:self.insertCur.execute("INSERT INTO "+self.schema_name+ ".dataset_details(dataset_id,parent_dataset_id,cmgdb_name,path_name,lfn,date_recorded,last_commented,file_owner,published_by) values(%d,NULL,'%s','%s','%s',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,'%s','%s')" % (datasetID,dsName,eos,lfn,fileOwner, username))
			else: self.insertCur.execute("INSERT INTO "+self.schema_name+ ".dataset_details(dataset_id,parent_dataset_id,cmgdb_name,path_name,lfn,date_recorded,last_commented,file_owner,published_by) values(%d,%d,'%s','%s','%s',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,'%s','%s')" % (datasetID,parentID,dsName,eos,lfn,fileOwner, username))
			self.insertConn.commit()
			return datasetID
		except cx_Oracle.IntegrityError:
			# If set is already in the database then print error message and ignore
			print "Dataset %s is already present on the system\n" % dsName
	
	# Adds a file group to the database
	def addFileGroup(self, group_name,cmgdb_id):
		"""Add a file group to a dataset
			
		group_name takes the name of the file group
		cmgdb_id takes the ID of the dataset to which the file group belongs
		"""
		try:
			group_id = self.getGroupIDWithDatasetAndName(group_name,cmgdb_id)
			if group_id is not None:
				return group_id
			#Insert information into database
			group_id = self.selectCur.execute("select " +self.schema_name+ ".file_group_id_seq.NEXTVAL from dual").fetchone()[0]
			self.insertCur.execute("INSERT INTO " +self.schema_name+ ".file_group_details(file_group_id,dataset_id,file_group_name) values(%d,%d,'%s')" % (group_id,cmgdb_id, group_name))
			self.insertConn.commit()
			return group_id
		except cx_Oracle.IntegrityError:
			# If set is already in the database then print error message and ignore
			print "-File group %s is already present on the system-\n" % group_name
	
	# Get a group ID with the group's name and the dataset ID
	def getGroupIDWithDatasetAndName(self,group_name,cmgdb_id):
		"""Returns the unique ID of a file group in CMGDB
			
		group_name takes the name of the file group
		cmgdb_id takes the unique ID of the dataset to which the file group belongs"""
		try:
			return self.selectCur.execute("SELECT file_group_id from "+self.schema_name+ ".file_group_details where file_group_name='%s' and dataset_id=%d" % (group_name, cmgdb_id)).fetchone()[0]
		except:
			return None
	
	# Closes a dataset on the database
	def closeDataset(self, dsName):
		"""Change the dataset_open field of the dataset_details table to 'N' for given dataset name.
			
		'dsName' takes dataset name in CMGDB format e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		Returns None
		"""
		try:
			# Insert information into database
			datasetID = self.insertCur.execute("update " +self.schema_name+ ".dataset_details set dataset_is_open='N' where cmgdb_name = '%s'" % dsName)
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			#If set is already in the database then print error message and ignore
			print "Dataset %s failed to be closed\n" % dsName
	
	# Add a Dataset fraction to a logged dataset
	def addPrimaryDatasetFraction(self,datasetID, fraction):
		"""Update dataset_fraction relating to the given CMGDB Dataset ID in CMGDB table dataset_details, to new given value.
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'fraction' takes the dataset size as a fraction of the primary dataset as a float
		Returns None
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set dataset_fraction='%f' WHERE dataset_id='%d'" % (fraction, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist"
		except TypeError:
			pass
	
	# Add the fraction of the primary dataset used to a group
	def addGroupPrimaryDatasetFraction(self,group_id, fraction):
		"""Update dataset_fraction relating to the given CMGDB Dataset ID in CMGDB table dataset_details, to new given value.
			
		'group_id' takes the unique ID of the file group
		'fraction' takes the dataset size as a fraction of the primary dataset as a float
		Returns None
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".file_group_details set dataset_fraction='%f' WHERE file_group_id='%d'" % (fraction, group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist"
		except TypeError:
			pass
	
	# Add a Dataset size to a logged dataset
	def addDatasetSize(self,group_id, size):
		"""Update field dataset_size_in_tb relating to the given CMGDB Dataset ID in CMGDB table dataset_details, to new given value.
			
		'group_id' takes the unique ID of the file group
		'size' takes the dataset size in TB dataset as a float
		Returns None
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".file_group_details set dataset_size_in_tb='%f' WHERE file_group_id='%d'" % (size, group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist"
		except TypeError:
			pass
	
	# Add a Directory size to a logged dataset
	def addDirectorySize(self,datasetID, size):
		"""Update field dataset_size_in_tb relating to the given CMGDB Dataset ID in CMGDB table dataset_details, to new given value.
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'size' takes the dataset size in TB dataset as a float
		Returns None
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set dataset_size_in_tb='%f' WHERE dataset_id='%d'" % (size, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist"
		except TypeError:
			pass
	
	# Add a Dataset file entries to a logged dataset
	def addFileEntries(self,datasetID, entries):
		"""Update dataset_entries relating to the given datasetID in CMGDB table dataset_details, to new given value.
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'entries' takes the number of file entries in the dataset as an int
		Returns None
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set dataset_entries='%d' WHERE dataset_id='%d'" % (int(entries), datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist"
		except TypeError:
			pass
	
	# Add the number of file entries to a group
	def addGroupFileEntries(self,group_id, entries):
		"""Update dataset_entries relating to the given datasetID in CMGDB table dataset_details, to new given value.

		'group_id' takes the unique ID of the file group
		'entries' takes the number of file entries in the dataset as an int
		Returns None
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".file_group_details set dataset_entries='%d' WHERE file_group_id='%d'" % (int(entries), group_id))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist"
		except TypeError:
			pass
	
	# Add a Dataset file entries to a logged dataset
	def addPrimaryDatasetEntries(self,datasetID, entries):
		"""Update dataset_entries relating to the given datasetID in CMGDB table dataset_details, to new given value.
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'entries' takes the number of file entries in the dataset as an int
		Returns None
		"""
		try:
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set primary_dataset_entries='%d' WHERE dataset_id='%d'" % (int(entries), datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist"
		except TypeError:
			pass
	
	# Get dataset ID with name
	def getDatasetIDWithName(self, datasetName):
		"""Return the unique CMGDB Dataset ID of a dataset with a given name.
			
		'datasetName' takes dataset name in CMGDB format e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		Return unique CMGDB Dataset ID of given dataset,
		Returns None if no dataset is found
		"""
		try:
			return self.selectCur.execute("SELECT dataset_id from " +self.schema_name+ ".dataset_details where cmgdb_name='%s'" % datasetName).fetchone()[0]
		except:
			return None
	
	# Return possible parents as a list
	def getParentsWithName(self, datasetName):
		"""Return a list of potential parents of a given dataset
			
		'datasetName' takes dataset name in CMGDB format e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		Returns list of possible parents (usually only 1) as a list of lists:
		-   [0] of each dataset (row) in the list holds the dataset name in CMGDB format e.g. /DiPhotonBox_Pt-250_7TeV-pythia6/Summer11-PU_S4_START42_V11-v1--V3---cmgtools/AODSIM
		-   [1] of each dataset (row) in the list holds the unique CMGDB Dataset ID of the dataset.
		If no parents found, returns an empty list e.g. []
			
		Function runs a regular expression search on all datasets,
		Search looks for datasets that match that match the parent dataset
		"""
		try:
			strippedDSName = re.sub("--[\w_]+---[\w_]+","---%",datasetName)
			datasets = self.selectCur.execute("SELECT cmgdb_name, dataset_id from " +self.schema_name+ ".dataset_details where cmgdb_name LIKE '%s' and dataset_is_open='Y'" % strippedDSName)
			checkedDatasets = []
			for row in datasets:
				check = re.sub("--[\w_]+---[\w_]+","---[\w_]+",datasetName)
				if re.match(check, row[0]):
					checkedDatasets.append(row)
			return checkedDatasets
		except:
			print "No Parent Dataset found"
			return []
	
	# Get dataset ID with DBS
	def getDatasetIDWithTaskID(self,taskID,test):
		"""Return unique CMGDB Dataset ID of the dataset with the given task ID
			
		'taskID' takes the Savannah task ID as an int
		'test' takes a boolean value of whether the task is a test task or not
		RETURN unique dataset id
		if not found return None
		### TODO: Colin, is test boolean needed? Seems obsolete now, could be removed to simplify code
		"""
                raise RuntimeError, "getDatasetIDWithTaskID called with %s" % [ taskID, test ]
		taskType = "task_id"
		# Insert information into database
		if test == True: taskType = "test_task_id"
		try:
			return self.selectCur.execute("SELECT dataset_id from " +self.schema_name+ ".dataset_details where %s=%d" % (taskType, int(taskID))).fetchone()[0]
		except: 
			print "Datset is new in CMGDB"
			return None
	
	# Method adds a Savannah Task ID to a dataset using the task_id attribute.
	def addTaskID(self, datasetID, taskID, test):
		"""Update columns in table dataset_details: last_commented and task_id of row pertaining to given dataset
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		'taskID' takes the Savannah task ID as an int
		'test' takes a boolean value of whether the task is a test task or not
		### TEST FIELD MAY BE SUBJECT TO REMOVAL
		"""
                raise RuntimeError, "addTaskID Called with %s" % [ datasetID, taskID, test ]
		try:
			taskType = "task_id"
			writtenType = "Task ID"
			#Insert information into database
			if test == True: 
				taskType = "test_task_id"
				writtenType = "Test task ID"
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set last_commented = CURRENT_TIMESTAMP WHERE dataset_id=%d" % int(datasetID))
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set task_id=%d WHERE dataset_id=%d" % (int(taskID),int(datasetID)))
			self.insertConn.commit()
			print "%s: %s added to Dataset ID: %d on CMGDB" % (writtenType,taskID,datasetID)
		except cx_Oracle.IntegrityError:
			# If set doesn't exist print error message and ignore
			print "Dataset doesn't exist on CMGDB archive\n"
		except TypeError:
			pass
	
	# Method adds a Tag Set ID to a dataset using the tagset_id attribute.
	def addTagSetID(self, tagsetID, datasetID):
		"""Update the column "tagset_id" in the table cms_cmgdb.dataset_details to the unique cmgdb tagset id of the given dataset
			
		'tagsetID' takes the unique CMGDB Tagset ID of the tagset as an int
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int
		Returns None
			
		Tagset id is a unique id which identifies a set of tags and a their CMSSW version
		This allows for easy comparison of the tags associated with different datasets, 
		and whether they are the same or not
		"""
		try:
			#Insert information into database (THIS IS THE LINE THAT WILL NEED TO CHANGE)
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set tagset_id='%d' WHERE dataset_id=%d" % (tagsetID, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			#If set doesn't exist print error message and ignore
			print "Dataset doesn't exist on CMGDB archive\n"
	
	# Returns the tag_ID value which can be used to map datasets to packages
	def getTagID(self, package, tag):
		"""Return the unique id of a tag
			
		'package' takes the name of the package as a string
		'tag' takes the name of the tag as a string
		Returns unique CMGDB tag ID as an int
			
		Allows you to lookup a tag id for use in searches and to add to new tagsets
		"""
		# Get the packages row using the select statement (tuple will only have 1 item, but will be returned as a tuple, not an int)
		row =  self.selectCur.execute("SELECT tag_id FROM " +self.schema_name+ ".tags WHERE package_name = :package AND tag=:tag",{"package":package,"tag":tag})
		# Return tuples first (and only) value
		try:
			return row.fetchone()[0]
		except Exception as ex:
			return None
	
	# Return task ID for given dataset
	def getTaskID(self, datasetID, test):
		"""Return the Savannah task ID related to the given dataset
			
		'datasetID' takes the unique CMGDB dataset ID of the dataset as an int
		'test' takes a boolean value determining whether the task ID you want is a test or not.
		Returns unique Savannah task ID as an int
		### TEST FIELD MAY BE SUBJECT TO REMOVAL
		"""
                raise RuntimeError, "getTaskID called with %s" % [ datasetID, test ]
		try:
			taskType = "task_id"
			# Insert information into database
			if test == True: taskType = "test_task_id"
			return self.selectCur.execute("SELECT %s from " +self.schema_name+ ".dataset_details where dataset_id='%d'" % (taskType, datasetID)).fetchone()[0]
		except:return None
	
	# Return an array of the tags
	def getTags(self, datasetID):
		"""Return a list of Dict objects, each containing a "Tag" field and a "Package" field, for all tags relating to a given dataset
			
		'datasetID' takes the unique CMGDB Dataset ID of the dataset as an int.
		Returns a list of Dict objects containing 2 fields:
		-   "Package" gives the name of the tags corresponding package as a string
		-   "Tag" gives the name of the tag as a string
		If no tags found, None is returned
		"""
		try:
			# Get tagset ID relating to dataset
			setID = self.selectCur.execute("Select tagset_id from " +self.schema_name+ ".dataset_details where dataset_id='%d'"% datasetID).fetchone()[0]
			tagDicts = []
			# Get a list of the tags in the tagset
			tags = self.selectCur.execute("select * from " +self.schema_name+ ".tags where tagset_id = '%d'" % setID)
			# Put the list of tags into Dict objects for easy formating
			for tag in tags:
				tagDicts.append({"Package":details[1],"Tag":details[2]})
			return tagDicts
		except: return None
	
	# Adds tags to tag set and set to dataset
	def addTagSet(self, release, hash):
		"""INSERT a new tagset into the CMGDB dataset, containing the release and a hash of all tags
			
		'release' takes the CMSSW release relating to the tagset as a string
		'hash' takes a hash of all the tags in the tagset as an int
		Returns unique CMGDB Tagset ID as an int
			
		First checks if a tagset with a matching hash already exists, if yes returns that tagset\'s CMGDB Tagset ID
		If no, adds the tagset to the database and returns the newly generated unique CMGDB Tagset ID
		"""
		# Check if tagset already exists, if yes returns ID
		tagSetID = self.getTagSetID(hash)
		if tagSetID is not None:
			return tagSetID
		# Get new unique id from set_id_seq and insert into database
		tagSetID = self.selectCur.execute("select " +self.schema_name+ ".set_id_seq.NEXTVAL from dual").fetchone()[0]
		self.insertCur.execute("insert into " +self.schema_name+ ".tag_sets values(:tagSetID, :release, :hash)", {"tagSetID":tagSetID,"release":release,"hash":hash})
		try:            
			self.insertConn.commit()
			return tagSetID
		except:
			print "Database error encountered, exiting"
			print "Tagset not added to dataset"
			return None
	
	# Add tags to a tag set
	def addTagToSet(self, tagID, tagSetID):
		""" Add a tag to a tagset by adding the unique CMGDG ID's of both the tagset and the tag to the table cms_cmgdb.tags_in_sets
			
		'tagID' takes the unique CMGDB Tag ID as an int
		'tagSetID' takes the unique CMGDB Tagset ID as an int
		"""
		try:
			self.insertCur.execute("insert into " +self.schema_name+ ".tags_in_sets values(:setID,:tag)",{"setID":tagSetID,"tag":tagID})
		except cx_Oracle.IntegrityError:
			print "Tag already in set"
		self.insertConn.commit()
		return tagSetID
	
	### get tag set ID with hash
	def getTagSetID(self, hash):
		"""Check if tagset already exists in system using a hash of the related tags, if yes, return its unique ID
			
		'hash' takes a hash of all the hashes of all the tags in the tagset as an int
		Returns tagset's unique CMGDB Tagset ID
		if no tagset found, None is returned
		"""
		try:
			set = self.selectCur.execute("select tagset_id from " +self.schema_name+ ".tag_sets where tag_hash=%d" % hash).fetchone()
			return set[0]
		except:
			return None

    	def addValidationRun( self, tagSetID, numberOfEventsProcessed ):
        	validationRunID = self.selectCur.execute("select " +self.schema_name+ ".validation_run_id_seq.NEXTVAL from dual").fetchone()[0]
        	try:
            		self.insertCur.execute("INSERT INTO " +self.schema_name+ ".validation_run_details(validation_run_id, validation_run_tagset_id, events_processed) values(%d, %d, %d)" % (validationRunID, tagSetID, int(numberOfEventsProcessed)))
            		self.insertConn.commit()
        	except cx_Oracle.IntegrityError:
            		#If exception is thrown display error message and ignore
            		print 'Validation run not added to validation_run_details'
            		exit( -1 )
        	return validationRunID

    	def addAnalyzer( self, analyzerName ):
        	try:
            		analyzerID = self.selectCur.execute("select analyzer_id from " +self.schema_name+ ".analyzers where analyzer_name='%s'" % analyzerName).fetchone()[0]
        	except:
            		analyzerID =  self.selectCur.execute("select " +self.schema_name+ ".analyzer_id_seq.NEXTVAL from dual").fetchone()[0]
            		try:
                		self.insertCur.execute("INSERT INTO " +self.schema_name+ ".analyzers(analyzer_id, analyzer_name) values(%d, '%s')" % (analyzerID,
                                                                                                                                analyzerName))
                		self.insertConn.commit()
            		except cx_Oracle.IntegrityError:
            			#If exception is thrown display error message and ignore
                		print 'ERROR - Analyzer not added to analyzers table'
                		print sys.exc_info()[1]
                		exit(-1)
        	return analyzerID

    	def addRootFilesToValidationRunWithAnalyzerOnDataset( self, validationRunID, eosPath, dir_name, analyzer_name, root_file ):
        	try:
            		lfnPath = eostools.eosToLFN(eosPath)
            		datasetInfo = self.selectCur.execute( "SELECT dataset_id,file_owner, path_name from " +self.schema_name+ ".dataset_details where lfn='%s'" % lfnPath 
).fetchone()
        	except:
            		print "ERROR - Could not find parent dataset of the analysis in CMGDB"
            		exit( -1 )

        	try: 
            		self.insertCur.execute( "INSERT INTO " +self.schema_name+ ".validation_root_files_details( validation_run_id, dataset_id, dir_name,  analyzer_name, root_file_name ) values(%d, %d, '%s', '%s', '%s')" % (validationRunID, datasetInfo[0], dir_name, analyzer_name, root_file ) )
            		self.insertConn.commit()
            		return datasetInfo
        	except cx_Oracle.IntegrityError:
            		print 'ERROR - Root file information not added to database'
            		print sys.exc_info()[1]
            		exit(-1)
        
    	def getValidationRunDetailsFromName( self, dir_name ):
        	try:
            		validationRunDetails = self.selectCur.execute( "SELECT validation_run_id, events_processed from " +self.schema_name+ ".validation_run_details where dir_name='%s'" % dir_name ).fetchone()
        	except:
            		print "ERROR - Validation run not found in database"
            		return None
        	return validationRunDetails

    	def getRootFileInfoFromValidationRunID( self, validationID ):
        	try:
            		rootFileInfo = self.selectCur.execute( "SELECT dir_name, lfn, analyzer_name, root_file_name FROM " +self.schema_name+ ".validation_root_files_details, analyzers, dataset_details WHERE dataset_details.dataset_id = validation_root_files_details.dataset_id AND validation_root_files_details.analyzer_id = analyzers.analyzer_id AND validation_run_id='%d'" % validationID ).fetchall()
        	except:
            		print "ERROR - Validation run id not found"
            		return None
        	return rootFileInfo

        def setParentID( self, datasetID, parentID ):
                """ re-set parent id when re-publishing """
                try:
			#Insert information into database (THIS IS THE LINE THAT WILL NEED TO CHANGE)
			self.insertCur.execute("UPDATE " +self.schema_name+ ".dataset_details set parent_dataset_id='%d' WHERE dataset_id=%d" % (parentID, datasetID))
			self.insertConn.commit()
		except cx_Oracle.IntegrityError:
			#If set doesn't exist print error message and ignore
			print "Dataset doesn't exist on CMGDB archive\n"

            

        
