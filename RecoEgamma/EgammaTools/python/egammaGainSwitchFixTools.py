import FWCore.ParameterSet.Config as cms 


def replaceInputTagModuleLabel(tag,names_dict):
     if tag.getModuleLabel() in names_dict:
   #      print "replacing",tag.getModuleLabel(),"with ",names_dict[tag.getModuleLabel()]
         tag.setModuleLabel(names_dict[tag.getModuleLabel()])

def checkTag(tag,names_dict):
    new_names=[names_dict[key] for key in names_dict]
    if tag.getModuleLabel() in new_names and tag.getProcessName()!="@skipCurrentProcess":
        return False
    return True
         
def replaceInputTags(process,modname,pset,names_dict):
    for paraname in pset.parameterNames_():
        para = pset.getParameter(paraname)
        if para.pythonTypeName()=="cms.PSet":
            replaceInputTags(process,modname,para,names_dict)
        elif para.pythonTypeName()=="cms.VPSet":
            for newpset in para:
                replaceInputTags(process,modname,newpset,names_dict)
        elif para.pythonTypeName()=="cms.InputTag":
            if not checkTag(para,names_dict):
                print "WARNING: {0}.{1} : {2} does not properly ignore the current process".format(modname,paraname,para.getModuleLabel())
            replaceInputTagModuleLabel(para,names_dict)
            
        elif para.pythonTypeName()=="cms.VInputTag":
            for tag in para:
                if not checkTag(tag,names_dict):
                    print "WARNING: {0}.{1}  does not properly ignore the current process".format(modname,paraname,tag.getModuleLabel())
                if tag.getModuleLabel() in names_dict:  
                    replaceInputTagModuleLabel(tag,names_dict)
                    
def replaceModulesInSeq(process,seq,names_dict):
    for org_name in names_dict.keys():
        seq.replace(getattr(process,org_name),getattr(process,names_dict[org_name]))
        


def customiseForAODGainSwitchFix(process,newNameSuffex=""):
    process.load("RecoEgamma.EgammaTools.egammaGainSwitchFix_cff")
    names_dict = {"ecalMultiAndGSGlobalRecHitEB" : "reducedEcalRecHitsEB" + newNameSuffex,
                  "particleFlowRecHitECALGSFixed" : "particleFlowRecHitECAL" + newNameSuffex,
                  "particleFlowRecHitPSGSFixed" : "particleFlowRecHitPS" + newNameSuffex,
                  "particleFlowClusterPSGSFixed" : "particleFlowClusterPS" + newNameSuffex,
                  "particleFlowClusterECALUncorrectedGSFixed" : "particleFlowClusterECALUncorrected" + newNameSuffex,
                  "particleFlowClusterECALGSFixed" : "particleFlowClusterECAL" + newNameSuffex,
                  "particleFlowSuperClusterECALGSFixed" : "particleFlowSuperClusterECAL" + newNameSuffex,
                  "gsFixedRefinedSuperClusters" : "particleFlowEGamma" + newNameSuffex,
                  "gsFixedGsfElectronCores" : "gedGsfElectronCores" + newNameSuffex,
                  "gsFixedGsfElectrons" : "gedGsfElectrons" + newNameSuffex,
                  "gsFixedGedPhotonCores" : "gedPhotonCore" + newNameSuffex,
                  "gsFixedGedPhotons" : "gedPhotons" + newNameSuffex}
    
 
    

    for org_name in names_dict.keys():
        setattr(process,names_dict[org_name],getattr(process,org_name).clone())        
     

    for seqname in process.sequences:
        seq = getattr(process,seqname)
        replaceModulesInSeq(process,seq,names_dict)
    
    mods_done=[] #keeps track of which modules we have done
                 #useful as module might be in multiple paths
    for pathname in process.pathNames() :
        try:
            path = getattr(process,pathname)
            for modulename in path.moduleNames():
                if modulename not in mods_done:
                    mod = getattr(process,modulename)
                    replaceInputTags(process,modulename,mod,names_dict)
                    mods_done.append(modulename)
        except AttributeError:
            pass
