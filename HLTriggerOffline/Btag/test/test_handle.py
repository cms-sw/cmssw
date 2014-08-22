#! /usr/bin/env python


import sys
import os
sys.argv.append('-b-')
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv.remove('-b-')

ROOT.gSystem.Load("libFWCoreCommon.so")
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.AutoLibraryLoader.enable()

products={
"hltBLifetimeL25JetsHbb":"std::vector<reco::CaloJet>",
"hltBLifetimeL25JetsbbPhi": "std::vector<reco::CaloJet>",
"hltBLifetimeL25BJetTagsHbb":"edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
"hltBLifetimeL3BJetTagsHbb":"edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
"hltBLifetimeL25TagInfosHbb":"std::vector<reco::TrackIPTagInfo>",
"hltBLifetimeL3TagInfosHbb":"std::vector<reco::TrackIPTagInfo>"

}


try:
 cmssw =os.environ["CMSSW_BASE"]
 print cmssw

except KeyError:
 print "Please ini CMSSW"
 print "We neend CMSSW for scipy!"
 sys.exit(1)


from DataFormats.FWLite import Events, Handle

file="output.root"
events = Events (file)

# create handle outside of loop
#handle  = Handle ('std::vector<reco::TrackIPTagInfo>')
handles=[]

# a label is just a tuple of strings that is initialized just
# like and edm::InputTag
#label = ("hltBLifetimeL25TagInfosHbb","","HLTX")
labels=[]
for key, value in products.items():
 handles.append( Handle (value))
 labels.append((key))

# loop over events
evt=0
for event in events:
 print "Event %d"%evt
#Let's get the data as we would in FWLite and cmsRun: 
 for i in range(len(handles)):
  try:

# use getByLabel, just like in cmsRun
   event.getByLabel (labels[i], handles[i])
   print "Processing %s"%labels[i]
   prod= handles[i].product()
  
#   try to get useful info
   if (labels[i]=="hltBLifetimeL25TagInfosHbb" or labels[i]=="hltBLifetimeL3TagInfosHbb" ):
    print "trying to extract taginfos : ", prod
    for tags  in prod:
     jet=tags.jet()
     print "extracted jet: ", jet
     print "jet pt is ",  jet.pt()
     tagvars=tags.taggingVariables()
     print "tagvar=", tagvars
     print "var=", tags.taggingVariables().get(0)
#     for var in tagvars.getList(22):
#      print "tag is ", var
     prod2=None
     if (labels[i]=="hltBLifetimeL25TagInfosHbb"): 
      print "Getting JetTags "
      prod2=Handle(products["hltBLifetimeL25BJetTagsHbb"])
      event.getByLabel ("hltBLifetimeL25BJetTagsHbb", prod2)
      print prod2
#     if (labels[i]=="hltBLifetimeL3TagInfosHbb"): prod2=event.getByLabel ("hltBLifetimeL3BJetTagsHbb", Handle(products["hltBLifetimeL3BJetTagsHbb"]))
     if (prod2 != None) : 
      print "Found valid ", prod2
      print prod2.value(0)
#      for k in len(prod2.size()):
#        print "element ", k
#        print prod2[k]
#       print "jet with pt ", prod3.first, " and tag ", prod3.second

  except:
   print "Something wrong with %s of type %s"%(labels[i], handles[i]) 
  

 evt=evt+1

