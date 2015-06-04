#!/usr/bin/env python
import os

if __name__ == "__main__":
    filename = "iov247026.txt"
    filehandler = open(filename, 'r')
    lines = filehandler.readlines()

    
    dirname = 'out_247026'
    try:
        os.stat(dirname)
    except:
        os.mkdir(dirname)
                
    recordname = None
    

    for line in lines:

        
        if '--- record' in line:
            recordname = line.split()[2]
            print 'record: ',recordname
            
        if 'prepMetaData' in line:
            dict_text = line.split('value: ')[1]
            print dict_text.replace('&quot;','"')
            outFilePrep = open('%s/%s_prep.json'%(dirname,recordname), 'w')
            outFilePrep.write(dict_text.replace('&quot;','"'))
            outFilePrep.close()
            
        if 'prodMetaData' in line:
            dict_text = line.split('value: ')[1]
            print dict_text.replace('&quot;','"')
            outFileProd = open('%s/%s_prod.json'%(dirname,recordname), 'w')
            outFileProd.write(dict_text.replace('&quot;','"'))
            outFileProd.close()
        

    filehandler.close()
        #print line
        
        
