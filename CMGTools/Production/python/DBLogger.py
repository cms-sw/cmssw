import os, re, datetime
import CMGTools.Production.eostools as castortools

from DBSAPI.dbsApiException import *


class DBLogger:
    def __init__(self, dirLocalOrTgzDirOnCastor, castorTgz, dbsAPI):
        #self.dbAPI = DatabaseAPI.DatabaseAPI('/afs/cern.ch/user/p/pmeckiff/public/bookkeeping.db')
        self.dirLocal = None
        self.tgzDirOnCastor = None
        self.dirOnCastor = None
        self.setName = dirLocalOrTgzDirOnCastor
        self.dbsAPI = dbsAPI

        
        # Set Directory name if local
        local = dirLocalOrTgzDirOnCastor.rstrip('/')
        castorTgz = castortools.castorToLFN(castorTgz)

        print castorTgz
        # Check if local first (obviously)
        if self.isDirLocal(local  ):
            print "File is on local machine: " + local
            self.dirLocal = local #if found set class attribute
        # Check if on castor next
        elif self.isTgzDirOnCastor(castorTgz):
            print "File is directory on Castor"
            self.tgzDirOnCastor = castorTgz # if found set class attribute
            for i in castortools.matchingFiles(castorTgz.rstrip("/Logger.tgz"), ".*tgz"): print i
        # If logger is not present but directory exists
        elif self.isDirOnCastor(castorTgz.rstrip("/Logger.tgz")):
            print "Directory is valid on Castor, but no logger file is present."
            self.dirOnCastor = castorTgz.rstrip("/Logger.tgz")
        # If neither then raise an exception
        else:
            raise ValueError( dirLocalOrTgzDirOnCastor + ' is neither a tgz directory on castor (provide a LFN!) nor a local directory')

    # Method for checking if file exists locally
    def isDirLocal(self, file ):
        if os.path.isdir( file ):
            return True
        else:
            return False

    # Method for checking if file exists on Castor
    def isTgzDirOnCastor(self, file ):
        if castortools.fileExists( file ):
            return True
        else:
            return False
    def isDirOnCastor(self, file ):
        if castortools.fileExists( file):
            return True
        else:
            return False

    # Stage in the Logger.tgz file, and open it inside local, newly created tempLogs folder
    def stageIn(self):

        # If tempLogs exists, remove it
        os.system("rm -r tempLogs")

        # Create a new tempLogs directory
        setup = 'mkdir tempLogs'
        os.system( setup )

        # If the file is a tgz directory on castor stage it in
        if self.tgzDirOnCastor != None:
            cmsStage = 'cmsStage -f ' + self.tgzDirOnCastor + ' ./tempLogs'
            print cmsStage 
            os.system( cmsStage ) 

            # Go to tempLogs folder and unzip Logger.tgz 
            os.chdir("tempLogs/")
            os.system('tar -zxvf ' + "Logger.tgz" )
            os.system('rm ' + "Logger.tgz" )
            os.chdir("../")

        # If file is on local disk, copy it into the tempLogs folder
        elif self.dirLocal != None:
            os.system("mkdir tempLogs/Logger")
            copyLoader = "cp " + self.dirLocal.rstrip("/") + "/Logger/*" + " tempLogs/Logger/"
            os.system(copyLoader)

        elif self.dirOnCastor != None:
            os.system("mkdir tempLogs/Logger")
            os.system("touch tempLogs/Logger/logger_showtags.txt")

        # Otherwise an error has occured so throw exception
        else:
            raise ValueError( 'cannot stage in, directory is invalid')

    # Delete templogs
    def stageOut(self):
          
            os.system('rm -r tempLogs')
            print 'Successfully staged out'


    # Add the dataset details
    #### not in use yet
    def addDataset(self, dataset):
        
        name = dataset['PathList'][0]
        args = name.lstrip("/").split("/")
        tiers = name.lstrip("/").lstrip(args[0]+"/"+args[1])

        #procs = len(datasets = self.dbsAPI.listProcessedDatasets(args[0], tiers, args[1]))  == 0
        procs = 0
        dbsID = None

        #if procs == 0:
            #try:
                #self.dbsAPI.insertProcessedDataset (dataset)

                #print "Result: %s" % proc
                #dbsID = int( self.dbsAPI.executeQuery(query="find procds.id where dataset="+name, type='exe').split("results>")[1].split("procds.id>")[1].rstrip("</"))

            #except DbsApiException, ex:
                #print "Caught API Exception %s: %s "  % (ex.getClassName(), ex.getErrorMessage() )
                #if ex.getErrorCode() not in (None, ""):
                    #print "DBS Exception Error Code: ", ex.getErrorCode()

        ##### NEW CODE

        #if procs == 0 and dbsID != None:
            #self.dbAPI.addSetDetails(self.setName, dbsID)

    # Checks contiguity of root files
    def checkRootType(self, name):
        suffix = []
        suffix = name.rstrip(".root").split("_")
        grid = False
        # 1st test, if passed, sample provisionally from grid
        try:
            tester = int(suffix[-2])
            tester = int(suffix[-3])
            grid = True
        except:
            grid = False
        # if failed sample is definitely grid, however some samples from grid will pass. (Hence first test)
        try:
            tester = int(suffix[-1])
        except:
            return True
        
        return grid
    
    # Turns a grid name into a name that is easy to compare. (helper method)
    def _stdNameFromGrid(self,name):
        
        return name.split("_"+name.split("_")[-3]+ "_"+name.split("_")[-2]+"_"+name.split("_")[-1])[0]

    def _checkIfNamed(self, name):
        suffix = []
        suffix = name.rstrip(".root").split("_")
        A =False
        B =False
        try: tester = int(suffix[-1])
        except: A = True
        try: tester = int(suffix[-2])
        except: B=True

        if A and B:
            return True
        else: return False
        
    def checkContiguity(self, targetDir):
        #GET ALL ROOT NAMES
        fileNames = castortools.matchingFiles(targetDir, ".*root")
        
        fileGroups =[]
        groupName = []
        # Loop while there are still filenames that do not belong to a file group
        while len(fileNames)>0:

            # Set filename for this pass as the current first element of the filename array
            filename = fileNames[0]

            # Create a new array to hold the names of the group
            fileGroup = []

            # Strip every filename (temporarily) of its file type, number and leading underscore, so that files from each
            # group (root set) have the same name
            for listItem in fileNames:
                # If names are of the same type (prevents a lot of unneccesary processing)
                if self.checkRootType(listItem) == self.checkRootType(filename):

                    # If item is from grid
                    if self.checkRootType(listItem):
                        #If items are the same
                        if self._stdNameFromGrid(listItem)==self._stdNameFromGrid(filename):
                            #print listItem
                            fileGroup.append(listItem)
                            

                    # If item is not from grid
                    elif listItem.rstrip("_[1234567890]*\.root")==filename.rstrip("_[1234567890]*\.root"):
                        # If the file name matches that of the first element in the fileNames array, they are of the same
                        # file group, so add to the fileGroup array
                        fileGroup.append(listItem)
                    

            # Remove the filenames that have been grouped, from the original filenames array,
            # so they do not get processed twice
            for item in fileGroup:
                fileNames.remove(item)

            # Add the new fileGroup to the array of fileGroups
            fileGroups.append(fileGroup)
            

        # Define a flag variable to check for incontiguous root sets
        groupFlag = True
        setFlag = True
        validity = []
        # Count through the groups
        for group in fileGroups:
            
            # Set name of group to be returned
            groupName = ""
            
            # Set an array for numbers
            numbers = []

            
            if self._checkIfNamed(group[0]):
                validity.append(group[0])
                print group[0]
            else:
                # Exract the filenumber from each file in the group and add it to the numbers array
                if self.checkRootType(group[0]):
                    for element in group:
                        num = element.split("_")[-3]
                        numbers.append(int(num))
                else:

                    for element in group:
                        num = element.rstrip(".root").split("_")[-1]
                        numbers.append(int(num))

                count = 0
                # Sort Numbers so that they are in ascending order
                numbers.sort()
                if numbers[0] == 1: count +=1
                # Check that all numbers are there and index every element
                for i in numbers:
                    # If an element is erroneous call up a flag and move on to the next set
                    if i != count:
                        groupFlag = False


                    count+=1
                # Create names for file groups
                if self.checkRootType(group[0]):
                    # Create name for grid type in format: name_[a-n]_identifier_XXX.root
                    arr = group[0].split("_")
                    arr[-1] = "XXX.root"
                    arr[-3] = "["+str(numbers[0])+"-" + str(numbers[-1])+"]"
                    groupName = "_".join(arr)
                    print groupName
                else:
                    # Create name for normal type in format name_[a-n].root
                    groupName = group[0].rstrip(str(numbers[0])+".root") +"["+str(numbers[0])+"-"+ str(numbers[-1])+"].root"
                    print groupName


                # Append group name with contiguity to return array
                if groupFlag==True:
                    validity.append(groupName+": CONTIGUOUS")
                else:
                    validity.append(groupName+": NON-CONTIGUOUS")
                    setFlag = False


        # If there are non-contiguous file sets, return false and print error message.
        # Otherwise return true

        if setFlag==False:
            print "There are non-contigious root files"
            validity.append("INVALID")
            
        else:
            print "Root files are all contiguous"
            validity.append("VALID")
        return validity
            
    #def __del__(self):
        #self.dbAPI.close()
