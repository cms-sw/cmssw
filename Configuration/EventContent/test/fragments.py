import FWCore.ParameterSet.Config as cms
import os
import pickle

def _yellow(string):
    return '%s%s%s' %('\033[1;33m',string,'\033[1;0m')  

def include(includes_set):
    """
    It takes a string or a list of strings and returns a list of 
    FWCore.ParameterSet.parseConfig._ConfigReturn objects.
    In the package directory it creates ASCII files in which the objects are coded. If 
    the files exist already it symply loads them.
    """
    
    func_id='[fragments.include]'
        
    #packagedir=os.environ["CMSSW_BASE"]+"/src/Configuration/PyReleaseValidation/data/"
    packagedir='./'
    #Trasform the includes_set in a list
    if not isinstance(includes_set,list):
        includes_set=[includes_set]
    
    object_list=[]    
    for cf_file_name in includes_set:
        pkl_file_name=packagedir+os.path.basename(cf_file_name)[:-4]+".pkl"
        
        cf_file_fullpath=""
        # Check the paths of the cffs
        for path in os.environ["CMSSW_SEARCH_PATH"].split(":"):
            cf_file_fullpath=path+"/"+cf_file_name
            if os.path.exists(cf_file_fullpath):
                break
        
        pkl_file_exists=os.path.exists(pkl_file_name)               
        # Check the dates of teh cff and the corresponding pickle
        cff_age=0
        pkl_age=0
        if pkl_file_exists:
            cff_age=os.path.getctime(cf_file_fullpath)
            pkl_age=os.path.getctime(pkl_file_name)
            if cff_age>pkl_age:
                print _yellow(func_id)+" Pickle object older than file ..."
        
       
        if not pkl_file_exists or cff_age>pkl_age:
          obj=cms.include(cf_file_name)
          file=open(pkl_file_name,"w")
          pickle.dump(obj,file)   
          file.close()
          print _yellow(func_id)+" Pickle object for "+cf_file_fullpath+" dumped as "+pkl_file_name+"..."
        # load the pkl files.                       
        file=open(pkl_file_name,"r")
        object_list.append(pickle.load(file))
        file.close()
        print _yellow(func_id)+" Pickle object for "+cf_file_fullpath+" loaded ..."
    
    return object_list
