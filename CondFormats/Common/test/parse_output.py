#!/usr/bin/env python
import os
import json


if __name__ == "__main__":
    filename = "out_sqlite_v2.txt"
    dirname = 'out_sqlite_v3'
    
    filehandler = open(filename, 'r')
    lines = filehandler.readlines()

    

    try:
        os.stat(dirname)
    except:
        os.mkdir(dirname)
                
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
            outFilePrep = open('%s/%s_prep.json'%(dirname,recordname), 'w')
            outFilePrep.write(prep_metadata_dump+'\n')
            outFilePrep.close()
            
        if 'prodMetaData' in line:
            dict_text = line.split('value: ')[1]
            prod_metadata = json.loads(dict_text.replace('&quot;','"'))
            prod_metadata_dump = json.dumps(prod_metadata, sort_keys = True, indent = 4)
            print '----- prodMetaData:'
            print prod_metadata_dump
            outFileProd = open('%s/%s_prod.json'%(dirname,recordname), 'w')
            outFileProd.write(prod_metadata_dump+'\n')
            outFileProd.close()
        

    filehandler.close()
        #print line
        
        
