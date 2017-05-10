from os import environ
from os.path import exists, join

def findFileInPath(theFile):                                                                                                                               
        for s in environ["CMSSW_SEARCH_PATH"].split(":"):                                                                                                      
                attempt = join(s,theFile)                                                                                                                          
                if exists(attempt):                                                                                                                                
                        return attempt                                                                                                                                 
        return None
