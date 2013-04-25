import sys

from Configuration.AlCa.autoCond import aliases
import Configuration.StandardSequences.FrontierConditions_GlobalTag_cff

class GlobalTagBuilderException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
        
class GlobalTag:
    def __init__(self, inputGT = "", inputConnect = "", inputPfnPrefix = "", inputPfnPostfix = "", inputGTParams = []):
        if inputGTParams == []:
            self.gtParams = []
            localConnect = inputConnect
            if localConnect == "":
                # print "No connection string specified for the GT. Using the default one:", Configuration.StandardSequences.FrontierConditions_GlobalTag_cff.GlobalTag.connect
                localConnect = Configuration.StandardSequences.FrontierConditions_GlobalTag_cff.GlobalTag.connect.value()
                # raise GlobalTagBuilderException("Error: no connection string specified.")
            localGT = inputGT
            # Expand the alias name
            if localGT in aliases:
                localGT = aliases[localGT]
            if localGT.find("|") != -1 and localConnect.find("|") == -1:
                # Fill a connection string for each GT
                connect = localConnect
                for i in range(1,len(localGT.split("|"))):
                    localConnect += "|"+connect
            self.gtParams.append([localGT, localConnect, inputPfnPrefix, inputPfnPostfix])
        else:
            self.gtParams = inputGTParams
        # print self.gtParams
    def __or__(self, other):
        if self.checkPrefix(other.gtParams) != -1:
            raise GlobalTagBuilderException("Error: trying to add the same GT component type \""+other.gtParams[0][0].split("_")[0]+"\" twice. This is not supported.")
        if len(other.gtParams) > 1:
            raise GlobalTagBuilderException("Error: the second GT has already a list. This is not supported.")
        tempGTParams = list(self.gtParams)
        tempGTParams.append(other.gtParams[0])
        return GlobalTag(inputGTParams = tempGTParams)
    def __add__(self, other):
        index = self.checkPrefix(other.gtParams)
        if index != -1:
            tempGTParams = list(self.gtParams)
            tempGTParams[index] = other.gtParams[0]
            return GlobalTag(inputGTParams = tempGTParams)
        else:
            exceptionString = "Error: replacement of GT "+other.gtParams[0][0]+" not allowed. No matching prefix found in existing GT components. Available components are:\n"
            for comp in self.gtParams:
                exceptionString += comp[0] + "\n"
            raise GlobalTagBuilderException(exceptionString)
    def checkPrefix(self, inputGTParams):
        """ Compares two input GTs to see if they have the same prefix. Returns the index in the internal list of GTs of the match
        or -1 in case of no match. """
        if inputGTParams[0][0].find("_") == -1:
            print "Invalid GT name. It does not contain an _, it cannot be used for replacements."
            sys.exit(1)
        prefix = inputGTParams[0][0].split("_")[0]
        for i in range(0, len(self.gtParams)):
            if self.gtParams[i][0].split("_")[0] == prefix:
                return i
        return -1
    def buildString(self, index):
        outputString = ""
        # print "index =", index
        # print self.gtParams
        for elem in self.gtParams:
            outputString += elem[index]
            if elem != self.gtParams[len(self.gtParams)-1]:
                outputString += "|"
        return outputString
        # return outputString.strip("|")
    def gt(self):
        return self.buildString(0)
    def connect(self):
        return self.buildString(1)
    def pfnPrefix(self):
        return self.buildString(2)
    def pfnPostfix(self):
        return self.buildString(3)
