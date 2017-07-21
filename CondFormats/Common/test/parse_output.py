#!/usr/bin/env python
import os
import json


# documentation: https://twiki.cern.ch/twiki/bin/view/CMS/AlCaDBPCL#Drop_box_metadata_management
if __name__ == "__main__":
    # the input file is in text format, formatted as the outout of cmsRun ProduceDropBoxMetadata.py
    # the inoput file holds metadata for a list of workflows, prod/prep for each
    filenameinput = "last-iov-DropBoxMetadata_v5.1_express.db-f422b9d9589e65175b255acc01700f9103842a6e.log"

    # the .json files will be produced inside the specified directory
    # each .json file is the complete metadata for either prod or prep
    dirnameoutput = 'last-iov-DropBoxMetadata_v5.1_express'
    
    filehandler = open(filenameinput, 'r')
    lines = filehandler.readlines()

    
    try:
        os.stat(dirnameoutput)
    except:
        os.mkdir(dirnameoutput)
                
    recordname = None
    

    for line in lines:

        
        if '--- record' in line:
            recordname = line.split()[2]
            print '-----------------------------------------------------------------'
            print 'record: ',recordname
            
        if 'prepMetaData' in line:
            dict_text = line.split('value: ')[1]
            prep_metadata = json.loads(dict_text.replace('&quot;','"'))
            prep_metadata_dump = json.dumps(prep_metadata, sort_keys = True, indent = 4)
            print '----- prepMetaData:'
            print prep_metadata_dump
            outFilePrep = open('%s/%s_prep.json'%(dirnameoutput,recordname), 'w')
            outFilePrep.write(prep_metadata_dump+'\n')
            outFilePrep.close()
            
        if 'prodMetaData' in line:
            dict_text = line.split('value: ')[1]
            prod_metadata = json.loads(dict_text.replace('&quot;','"'))
            prod_metadata_dump = json.dumps(prod_metadata, sort_keys = True, indent = 4)
            print '----- prodMetaData:'
            print prod_metadata_dump
            outFileProd = open('%s/%s_prod.json'%(dirnameoutput,recordname), 'w')
            outFileProd.write(prod_metadata_dump+'\n')
            outFileProd.close()
        

    filehandler.close()
        #print line
