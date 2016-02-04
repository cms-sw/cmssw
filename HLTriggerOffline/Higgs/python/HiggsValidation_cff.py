
####  based on EgammaValidation_cff  

import FWCore.ParameterSet.Config as cms

class dummy:
    pass

samples=dummy()
files=dummy()

##########################################################
# Define which preselections to run                      #
##########################################################

samples.names = ['HWW',
                 'Hgg',
		 'H2tau',
		 'HZZ',
		 'Htaunu'
		 ]

files.HWW = ['HLTHiggsBits_WW']
files.Hgg = ['HLTHiggsBits_gg']
files.H2tau = ['HLTHiggsBits_2tau']
files.HZZ = ['HLTHiggsBits_ZZ']
files.Htaunu = ['HLTHiggsBits_taunu']

              	      

tmp = cms.SequencePlaceholder("tmp")
HiggsValidationSequence = cms.Sequence(tmp)  # no empty sequences allowed, start with dummy

#loop over samples
for samplenum in range(len(samples.names)):

   
    # loop over triggers for each sample
  
   for trig in getattr(files,samples.names[samplenum]):
       trigname = trig + samples.names[samplenum] 
        #import appropriate config snippet
       # filename = "HLTriggerOffline.Higgs."+trig+"_cfi"
       filename = "HLTriggerOffline.Higgs."+trig+"_cfi"
       trigdef =__import__( filename )
       import sys
       globals()[trigname] = getattr(sys.modules[filename],trig).clone()    # clone imported config
      #  setattr(globals()[trigname],"cutcollection",cms.InputTag(fiducialname))        # set preselacted generator collection
      #  setattr(globals()[trigname],"cutnum",cms.int32( samples.num[samplenum]  )) # cut value for preselection
       # setattr(globals()[trigname],"pdgGen",cms.int32( samples.pdgid[samplenum])) #correct pdgId for MC matching
       # getattr(globals()[trigname],'triggerobject').setProcessName( lumiprocess[pathlumi[trig]] )         #set proper process name
       # for filterpset in getattr(globals()[trigname],'filters'):
        #    getattr(filterpset,'HLTCollectionLabels').setProcessName( lumiprocess[pathlumi[trig]] )
         #   for isocollections in getattr(filterpset,'IsoCollections'):
          #      isocollections.setProcessName( lumiprocess[pathlumi[trig]])

       HiggsValidationSequence *= globals()[trigname]                      # add to sequence


HiggsValidationSequence.remove(tmp)  # remove the initial dummy
