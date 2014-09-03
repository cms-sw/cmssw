#!/usr/bin/env python
#
# import modules
import os, time, shutil, zipfile, commands, sys, glob
from datetime import datetime
# import ends
#

#
# global variables for zipping and file transfer
# Directory Setup
dir = "/nfshome0/smaruyam/CMSSW_2_0_10/src/test/" # File Directory
dbdir = "/nfshome0/smaruyam/CMSSW_2_0_10/src/test/" # db Directory
arcdir = "/nfshome0/smaruyam/CMSSW_2_0_10/src/test/" # Zipped File Directory
cfgfile = " /nfshome0/smaruyam/CMSSW_2_0_10/src/test/myconfig.txt "# configuration file
# Directory Setup over
# Switches to en/disable functionalities
EnableFileRemoval = False
PathReplace = False
EnableTransfer = False
# Switches over
fileSizeThreshold =  1000000000# = 1GB(default) to get away from technicality of large zip file size
disk_threshold = 80#default 80% full
transferScript = "/nfshome0/tier0/scripts/injectFileIntoTransferSystem.pl"# T0 System Script
targetdir = "/castor/cern.ch/cms/store/dqm/" # Castor Store Area
cfgarg  = " --config " + cfgfile
fullTransferArg = cfgarg + " --type dqm --hostname srv-C2D05-19 --lumisection 1 --appname CMSSW --appversion CMSSW_2_0_10 "
statusCheck = cfgarg + " --check --filename "
emptyString = "empty"

# temporary fix for sqlite3 path
sqlite3 = "sqlite3 "
#

# commond database path and log file
logfile = open('archival_log.txt', 'a')# log
tmpdb = dbdir + "tmp/tmp.db" # temporary db
bakdb = dbdir + "tmp/backup.db" # backup db
db = dbdir + "db.db" # db
# global variables definition ends
#

#
# file register and un-register
# You don't need copy these!
"""
Temporary Port form Hyunkwan's Un-Register-File Script
"""
def filereg(db,bakdb,tmpdb,file,logfile):
    if os.path.exists(tmpdb): os.remove(tmpdb)
    shutil.copy(db,tmpdb)
    logfile.write('*** File Register ***\n')
    logfile.write(os.popen('visDQMRegisterFile '+ tmpdb +' "/Global/Online/ALL" "Global run" '+ file).read())
    t = datetime.now()
    tstamp = t.strftime("%Y%m%d")
    a = glob.glob(bakdb+'.'+tstamp+'*');
    if not len(a):
        tstamp = t.strftime("%Y%m%d_%H%M%S")
        bakdb = bakdb+'.'+tstamp
        shutil.copy(tmpdb,bakdb)
        shutil.move(tmpdb,db)
    else:
        shutil.move(tmpdb,db)

def fileunreg(db,bakdb,tmpdb,oldfile,logfile):
    if os.path.exists(tmpdb): os.remove(tmpdb)
    shutil.copy(db,tmpdb)
    logfile.write('*** File UnRegister ***\n')
    logfile.write(os.popen('visDQMUnregisterFile '+ tmpdb +' ' + oldfile).read())
    t = datetime.now()
    tstamp = t.strftime("%Y%m%d")
    a = glob.glob(bakdb+'.'+tstamp+'*');
    if not len(a):
        tstamp = t.strftime("%Y%m%d_%H%M%S")
        bakdb = bakdb+'.'+tstamp
        shutil.copy(tmpdb,bakdb)
        shutil.move(tmpdb,db)
    else:
        shutil.move(tmpdb,db)

# file register and un-register over
#

#
# generic function
# check command exist status and retrive output messsage
"""
Check and Return Output
"""
def CheckCommand(cmd, logfile):
	result = commands.getstatusoutput(cmd)
	if result[0] == 0:
		output = result[1]
		return result
	else :
		logfile.write("Command Exits with non-zero Status," + str(result[0]) + " Error = " + result[1] + "\n")
		return result

# generic function over
#

#
# disk use check
"""
Disk Usage Check
Reference to Cleaner()
df out put is assumed as follows.
Filesystem            Size  Used Avail Use% Mounted on
/dev/sda3              73G   45G   25G  65% /
/dev/sda1              99M   12M   83M  12% /boot
none                  2.0G     0  2.0G   0% /dev/shm
/dev/sdb1             917G   83G  788G  10% /data
cmsnfshome0:/nfshome0
                      805G  673G  133G  84% /cmsnfshome0/nfshome0
"""
def DiskUsage(logfile) :
    logfile.write(" *** Checking Disk Usage ***\n")
    df_file=os.popen('df')
    usage = False
    lines = df_file.readlines()
    list = lines[4].split() # 5th line from top. Split at tab or white space
    string = list[4][:-1] # NEED check for the host
    fusage = float(string)
    if fusage > disk_threshold : # disk is more than 80% full
        logfile.write("Disk Usage too high = " + string + "%\n")
        usage = True
    if usage == True :
        Cleaner(logfile)
    else :
        logfile.write("Disk Usage is low enough = " + string + "%\n")

# disk use check over
#

#
# Confirm the path to Transferred file on Castor
"""
Set Path to Castor
Reference to CheckPath(), filereg()
"""
def SetPath(file, logfile):
	path = CheckPath(file,logfile)
	if cmp(path,emptyString) != 0:
		newpath = "rfio:" + path
		logfile.write("Register New Path " +  newpath + "\n")
		filereg(db,bakdb,tmpdb,newpath,logfile)
		return True
	else :logfile.write("File Transferred, but not found on tape\n")
	return False

"""
Path Specifier
Reference to ConfirmPath(), ScanDir()
"""
def CheckPath(filename, logfile) :
	mtime = os.stat(filename).st_mtime
	year = time.localtime(mtime)[0]
        month = time.localtime(mtime)[1]
	if month > 9: yearmonth = str(year) + str(month)
	else: yearmonth = str(year) + "0" + str(month)
        path = targetdir + yearmonth
	logfile.write("Best Guess for the path is " + path + "\n")# guess the path based on mtime
        newpath = ConfirmPath(filename, path, logfile)# check if the path is correct
        if cmp(newpath,emptyString) != 0 : return newpath
        else :# scan all path, if the guess is wrong
		newpath = ScanDir(filename, logfile)
		return newpath

"""
Check File Path on Tape
Reference to CheckCommand()
""" 
def ConfirmPath(file, path, logfile) :
	logfile.write(" *** Checking File Path ***\n ")
	time.sleep(10) 
	fullpath = path + "/" + file[len(arcdir):]
	mycmd = "rfdir "
	myarg = fullpath
	cmd = mycmd + myarg
	result = CheckCommand(cmd, logfile)
	if result[0] == 0:
		output = result[1]
		if cmp(output,"") != 0:
			for line in output.split("\n"):
				error_check = "No such file or directory"
				if line.find(error_check) != -1 :return emptyString
				logfile.write(" rfdir result is " + line + "\n")
				if len(line.split()) > 7:
					string = line.split()[-1]
					if cmp(string,fullpath) == 0: return fullpath
	return emptyString

"""
Scan Castor Directories
Reference to ConfirmPath(), CheckCommand()
"""
def ScanDir(file, logfile) :
        mycmd = "rfdir "
        myarg = targetdir
        cmd = mycmd + myarg
        logfile.write("Scanning tape area  " + cmd + "\n")
        result = CheckCommand(cmd, logfile)
        if result[0] == 0:
		if cmp(result[1],"") != 0:
	                output = result[1].split('\n')
        	        for line in output :
				if len(line.split()) > 8:
					newpath = targetdir + line.split()[-1]
					logfile.write("Looking for File at " + newpath + "\n")
					confirmpath = ConfirmPath(file, newpath, logfile)
					logfile.write("Returned Path " + confirmpath + "\n")
					if cmp(confirmpath, newpath + "/" + file[len(arcdir):] ) == 0: return confirmpath
	return emptyString

# path check over
#

#
# T0 transfer functions
"""
Transfer File with T0 System
Reference to CheckCommand()
"""
def TransferWithT0System(filepath, flag, logfile):
	filename = filepath[len(arcdir):]
	nrun = filepath[len(arcdir)+len("DQM_Online_R"):-len("_R000064807.zip")]# file name length matters!
	transfer_string = transferScript + " --runnumber " + nrun + " --path " + arcdir + " --filename " + filename
	if EnableTransfer is False: transfer_string += " --test "# TEST, no file transfer
	if flag is True: transfer_string += " --renotify "# transfer failed previously, trying to send it again
	mycmd = transfer_string
	myarg = fullTransferArg
	cmd = mycmd + myarg
	result = CheckCommand(cmd, logfile)
	if result[0] == 0:
		output = result[1].split('\n')
		for line in output:
        	        if line.find("File sucessfully submitted for transfer.") != -1 and flag is False:
                	        logfile.write("File is queued " + filepath + "\n")
                        	return True
        	        if line.find("File sucessfully re-submitted for transfer.") != -1 and flag is True:
                	        logfile.write("File is resubmitted " + filepath + "\n")
                        	return True
	if EnableTransfer is False: logfile.write(" *** Transfer Test Mode ***\n No File Transferred.\n") # TEST, no file transfer
	return False

"""
Check File Status of Transferred File 
Reference to CheckCommand(), TransferWithT0System()
"""
def CheckFileStatus(filepath, logfile):
        filename = filepath[len(arcdir):]
	checkString = statusCheck + filename
	mycmd = transferScript
	myarg = checkString
	cmd = mycmd + myarg
	result = CheckCommand(cmd, logfile)
	if result[0] == 0:
		output = result[1].split('\n')
		for line in output:
			if line.find("FILES_TRANS_CHECKED: File found in database and checked by T0 system.") != -1: return True# file transferred successfully!
			elif line.find("File not found in database.") != -1:# file not transferred at all
				flag = False
				TransferWithT0System(filepath,flag, logfile)
				return False
			elif line.find("FILES_INJECTED : File found in database and handed over to T0 system. ") != -1:# file must be transferred
				flag = True
				TransferWithT0System(filepath,flag, logfile)
				mtime = os.stat(filepath).st_mtime
				logfile.write("Old M Time is " + mtime + "\n")
				os.utime(filepath,None)# change mtime to help path search
				mtime2 = os.stat(filepath).st_mtime
				logfile.write("New M Time is " + mtime2 + "\n")
	logfile.write("File transfer need more time, please wait!\n")
	return False

# T0 transfer over
#

#
# read file list from db
"""
Get List of un-merged files, for Zipping
Reference to GetFileFromDB(), GetZippedFile()
"""
def GetListOfFiles(logfile):
	logfile.write("Retrieving list of files from DB ...\n")
	totalSize = 0
	zipFileList = ''
	if PathReplace is True:# file removal is involved
		fileList = GetFileFromDB(logfile).split('\n')
		for line in fileList:
			if cmp(line,"") != 0 and cmp(line,emptyString) != 0:
				string = line.rstrip().split('|')
				name = string[0]
				logfile.write("String just read is " + string + "\n")
				number = string[1]
				logfile.write("Number just read is " + number + "\n")
				totalSize += int(number)
				logfile.write("Current File Size Sum is " + str(totalSize) + " out of Limit" + str(fileSizeThreshold) + "\n")
				zipFileList += " " + name
				if totalSize > fileSizeThreshold:
        				return zipFileList
	if PathReplace is False:# file removal is NOT involved
		activate = False
		lastfile = ""
		flag = True
		mergedfiles = GetZippedFile(logfile,flag).split("\n")
		if len(mergedfiles) > 0:
			if cmp(mergedfiles[0],"") != 0 and cmp(mergedfiles[0],emptyString) != 0:
				lastfile = mergedfiles[0]
				logfile.write("Last Merged Zip File is " + lastfile + "\n")
			elif  cmp(mergedfiles[0],"") == 0:
				activate = True
				logfile.write("No Merged Zip File \n")
		if len(mergedfiles) == 0:
			activate = True
			logfile.write("No Merged Zip File \n")
		fileList = GetFileFromDB(logfile).split("\n")
		for line in fileList:
			if cmp(line,"") != 0 and cmp(line,emptyString) != 0:
				string = line.split('|')
				name = string[0]
				if activate is True:
					logfile.write("Name just read is " + name + "\n")
					number = string[1]
					logfile.write("Number just read is " + number + "\n")
					totalSize += int(number)
					logfile.write("Current File Size Sum is " + str(totalSize) + " out of Limit" + str(fileSizeThreshold) + "\n")
					zipFileList += " " + name
					if totalSize > fileSizeThreshold:
       						return zipFileList
				if activate is False and cmp(lastfile,"") !=0:
					if cmp(lastfile[len(arcdir)+len("DQM_Online_R000064821_"):-len(".zip")],name[len(dir)+len("DQM_V0001_R000064821_"):-len(".root")]) == 0:
						activate = True
	return emptyString # it's too small

"""
Read and sort file from db, for Zipping
Reference to CheckCommand()
"""
def GetFileFromDB(logfile):
	logfile.write(" *** Getting Per-Run File List from Master DB ***\n")
	string = "'%DQM_V%_R%.root'"
        search1 = "'%RPC%'"
        search2 = "'%zip%'"
        sqlite = " %s \"select name, size from t_files where name like %s and not name like %s and not name like %s order by mtime asc\" "  %(db, string, search1, search2)
	mycmd = sqlite3
	myarg = sqlite
	cmd = mycmd + myarg
        result = CheckCommand(cmd, logfile)
        if result[0] == 0:
                return result[1]
	else:
		logfile.write(result[1])
		return emptyString

"""
Get the last merged File, for Zipping
Reference to CheckCommand()
"""
def GetZippedFile(logfile, flag):
	logfile.write(" *** Getting Zipped File List from Master DB ***\n")
	string = "'%DQM%.zip'"
        if flag is True: sqlite = " %s \"select name from t_files where name like %s order by mtime desc\" "  %(db, string)
        if flag is False: sqlite = " %s \"select name from t_files where name like %s order by mtime asc\" "  %(db, string)
	mycmd = sqlite3
	myarg = sqlite
	cmd = mycmd + myarg
        result = CheckCommand(cmd, logfile)
        if result[0] == 0:
                return result[1]
	else: return emptyString

"""
Getting All Files from DB, for File Removal
Reference to CheckCommand()
"""
def GetAllFiles(logfile) :
	logfile.write(" *** Getting All Files from db ***\n")
        sqlite = db + " \"select name from t_files where name like '%DQM%.root' or name like '%DQM%.zip'order by mtime asc\""
	mycmd = sqlite3
	myarg = sqlite
	cmd = mycmd + myarg
	result = CheckCommand(cmd, logfile)
	if result[0] == 0:
		output = result[1].split('\n')
		return output
	else : return emptyString

# file list over
#

#
# remove files and register/unregister if needed
"""
File Cleaner, Remove the oldest file
Reference to GetAllFiles(), CheckFileStatus(), SetPath(), Delete(), CheckZippedFiles()
"""
def Cleaner(logfile) :
	logfile.write(" *** Cleaning File ***\n")
	files = GetAllFiles(logfile)
	for file in files:
		if file.find(".zip") != -1:#zip file
			status = CheckFileStatus(file, logfile)# check transfer status
			if status is True and PathReplace is True:# remove file and replace the place
				pathfind = SetPath(file, logfile)
				if pathfind is True :# path found on tape
					Delete(file, logfile)# remove only if transferred 
					return # exits when the files deleted
		if file.find(".root") != -1 and file.find(dir) != -1:# Select Per-Run files
			if PathReplace is False: CheckZippedFiles(file, logfile)# need check if zipped or not
			if PathReplace is True: Delete(file, logfile)# must be zipped by this step
			return # exits when the file deleted. ie, delete only one file
	else : logfile.write("No File to be removed!\n")

"""
Remove File if zipped
Reference to Delete()
"""
def CheckZippedFiles(file, logfile):
	logfile.write(" *** Check Zipped File ***\n")
	flag = False
	mergedfiles = GetZippedFile(logfile,flag).split("\n")
	if len(mergedfiles) > 0:
		for thisfile in mergedfiles:
			if thisfile.find("zip") != -1 and cmp(thisfile,"") != 0 and cmp(thisfile,emptyString) != 0:
				zip = zipfile.ZipFile(thisfile, "r")# open file to see it readable
				for info in zip.infolist():# to see zipfile is uncompressed
					if cmp(info.filename, file) == 0:
						Delete(file,logfile)
						return True
	logfile.write("This file hasn't been zipped, " + file + " It shouldn't be deleted now!\n")

"""
Remove and Register Files
Reference to Delete(), filereg()
"""
def RemoveAndRegister(newFile,oldFiles, logfile):
        for file in oldFiles.split():
		newpath = newFile + "#" + file[len(dir):]
		logfile.write("Registering New File Path " + newpath +"\n")
		filereg(db,bakdb,tmpdb,newpath,logfile)
                Delete(file, logfile)

"""
Remove and Unregister A File
Reference to fileunreg()
"""
def Delete(file, logfile):
	fileunreg(db,bakdb,tmpdb,file,logfile)
	logfile.write(file + "removed from db...\n")
	os.remove(file)
	logfile.write(file + "removed from disk...\n")

# removal over
#

#
# main program
"""
Main Prog
Reference to DiskUsage(), GetListOfFiles(), TransferWithT0System(), RemoveAndRegister()
"""
if __name__ == "__main__":
	logfile.write("Starting Archival *Test* Script ...\n")
	if EnableFileRemoval is True: DiskUsage(logfile)# check disk usage
	zipFileList = GetListOfFiles(logfile) # get list of files for merging
	if cmp(zipFileList, emptyString) == 0 : logfile.write("Sum of Files is below Threshold = " + str(fileSizeThreshold) + "\n")
	else :# make zip file only if the output file will be large enough
		firstFile = "DQM_Online_" + zipFileList.split()[0][len(dir)+len("DQM_V0010_"):-len("R000064807.root")]
		lastFile  = zipFileList.split()[-1][len(dir)+len("DQM_V0010_R000064807_"):-len(".root")]
		outputFileName = arcdir + firstFile + lastFile + ".zip"
		logfile.write("1st File = " + firstFile + " Last File = " + lastFile + "\n")
		if os.path.exists(outputFileName) is True: os.remove(outputFileName)# remove old one if exists
		if lastFile.find("R") != -1 and firstFile.find("R") != -1:
			zip = zipfile.ZipFile(outputFileName, "w")# create zip file
			for name in zipFileList.split():
				zip.write(name,name, zipfile.ZIP_STORED)# add each file
			zip.close()# close zip file
			filepath = outputFileName
			zipFileSize = os.path.getsize(filepath)
			logfile.write("Zip File Size = " + str(zipFileSize) + "\n") 
			if zipFileSize > fileSizeThreshold :# check if file is large enough 
				zip = zipfile.ZipFile(outputFileName, "r")# open file to see it readable
				for info in zip.infolist():# to see zipfile is uncompressed
					logfile.write("File = " + info.filename + "\n")
				zip.close()# close zip file
				if PathReplace is False: filereg(db,bakdb,tmpdb,filepath,logfile)
				flag = False# brand new transfer 
				transfer = TransferWithT0System(filepath,flag, logfile)# Sending file to Castor
				if transfer is True and PathReplace is True: RemoveAndRegister(filepath,zipFileList, logfile)# register newpaths and remove files
			else:
				logfile.write("Inconsistency! Created Zip File too small!\n")
				raise RuntimeError
		else:
			logfile.write("Wrong File Name Stripping! Check directory path to the file!\n")
			raise RuntimeError
	logfile.close()

# main program over
#
