#! /usr/bin/env python

import ROOT
import sys
from DataFormats.FWLite import Events, Handle
events = Events (['avtester.root'])

handle  = Handle ("edm::AssociationVector<edm::RefProd<std::vector<edmtest::Simple> >,std::vector<edmtest::Simple>,edm::Ref<std::vector<edmtest::Simple>,edmtest::Simple,edm::refhelper::FindUsingAdvance<std::vector<edmtest::Simple>,edmtest::Simple> >,unsigned int,edm::helper::AssociationIdenticalKeyReference>")

label = ("tester","","TEST")


# loop over events
count= 0
for event in events:
    #print "###################### ", count
    event.getByLabel (label, handle)
    cont = handle.product()
    values = [ cont.value(i).value for i in xrange(len(cont))]
    for i,v in enumerate(handle.product()):
      #print v.second.value, values[i]
      if v.second.value != values[i]:
        raise RuntimeError("Values do not match for event index:{0}  from data:{1} from ref:{2}".format(count, v.second.value,values[i]))
    count+=1 
