from __future__ import print_function
import xml.etree.ElementTree as ET
import sys

## declare all constants here
TARGET_LIST_OF_TAGS=['BeamSpotObject_ByLumi', 'BeamSpotObject_ByRun', 'BeamSpotObjectHP_ByLumi', 'BeamSpotObjectHP_ByRun',
                     'SiPixelQualityFromDbRcd_other', 'SiPixelQualityFromDbRcd_prompt', 'SiPixelQualityFromDbRcd_stuckTBM',
                     'SiStripApvGain_pcl', 'SiStripApvGainAAG_pcl',
                     'SiStripBadStrip_pcl', 'SiPixelAli_pcl']
TARGET_DQM_FILES=1
TARGET_DQM_FILENAME='./DQM_V0001_R000325022__Express__PCLTest__ALCAPROMPT.root'
TARGET_DB_FILES=11
TARGET_DB_FILENAME='sqlite_file:promptCalibConditions.db'
TOTAL_TARGET_FILES=TARGET_DQM_FILES+TARGET_DB_FILES

#_____________________________________________________
def parseXML(xmlfile):
  
    # create element tree object
    tree = ET.parse(xmlfile)
  
    # get root element
    root = tree.getroot()

    totAnaEntries=len(root.findall('AnalysisFile'))

    if(totAnaEntries!=TOTAL_TARGET_FILES):
        print("ERROR: found a not expected number (",totAnaEntries,") of AnalysisFile entries in the FrameworkJobReport.xml")
        return -1

    listOfInputTags=[]

    countDBfiles=0
    countDQMfiles=0

    # iterate news items
    for item in root.findall('AnalysisFile'):
        # iterate child elements of item
        for child in item:
            if(child.tag == 'FileName'):
                if(child.text==TARGET_DB_FILENAME):
                    countDBfiles+=1
                elif(child.text==TARGET_DQM_FILENAME):
                    countDQMfiles+=1
                else:
                    pass
            if(child.tag == 'inputtag'):
                listOfInputTags.append(child.attrib['Value'])

    if(countDBfiles!=TARGET_DB_FILES):
        print("ERROR! Found an unexpected number of DB files (",countDBfiles,")")
        return -1

    if(countDQMfiles!=TARGET_DQM_FILES):
        print("ERROR! Found an uexpected number of DQM files (",countDQMfiles,")")
        return -1

    ## That's strict! 
    if(listOfInputTags!=TARGET_LIST_OF_TAGS):
        if (listOfInputTags>TARGET_LIST_OF_TAGS):
            print("ERROR!\n This ",[x for x in TARGET_LIST_OF_TAGS if x not in listOfInputTags]," is the set of expected tags not found in the FwdJobReport!")
        else:
            print("ERROR!\n This ",[x for x in listOfInputTags if x not in TARGET_LIST_OF_TAGS]," is the set of tags found in the FwkJobReport, but not expected!")
        return -1
    
    return 0

#_____________________________________________________
def main():
    try:
        f = open("FrameworkJobReport.xml")
    except IOError:
        print("FrameworkJobReport.xml is not accessible")
        sys.exit(1)

    # parse xml file
    result = parseXML('FrameworkJobReport.xml')
    if(result==0):
        print("All is fine with the world!")
        sys.exit(0)
    else:
        print("Parsing the FwkJobReport results in failure!")
        sys.exit(1)

#_____________________________________________________
if __name__ == "__main__":
  
    # calling main function
    main()
