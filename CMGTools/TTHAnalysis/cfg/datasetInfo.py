#!/bin/env python

import os


class Dataset:
  def __init__(self, name, id, xsect, kfact=1., filt=1.):
    self.name=name;
    self.id=id;
    self.xsection=xsect;
    self.kfactor=kfact;
    self.filter=filt;

  def scale1fb(self, events):
    if self.id>100: # mc
      return self.xsection*self.kfactor*1000.*self.filter/events
    else: # data
      return 1.



class DatasetDict(dict):

    def __keytransform__(self, key):
        return key

    def __getitem__(self, key):
        try:
          key.split()
          key = key.replace("-", "")
        except:
          pass
        return dict.__getitem__(self, self.__keytransform__(key))

    def __setitem__(self, key, value):
        return dict.__setitem__(self, self.__keytransform__(key), value)

    def addDataset( self, d ):
        # giovanni's convention: remove all hyphens:
        d.name = d.name.replace('-', '')
        # add twice, so that both the dataset name AND the sample ID can be used as keys:
        dict.__setitem__(self, self.__keytransform__(d.id),   d)
        dict.__setitem__(self, self.__keytransform__(d.name), d)

        print "[DatasetDict] :: Added dataset :"
        print "         " + d.name
        print "         id: " + str(d.id) 
        print "         x-sect: " + str(d.xsection)
        print "         kfactor: " + str(d.kfactor)
        print "         filter: " + str(d.filter)
        print "------------------------------------------------------"


    def addDatasetsFromFile( self, filename ):

      if not os.path.isfile(filename):
        os.system("wget https://mangano.web.cern.ch/mangano/public/MECCA/" + filename)

      file = open(filename, "r") 

      print "[DatasetDict] :: Adding samples from file: " + filename
   
      for line in file:
        if line.startswith("#"): continue
        args = line.split()
        if len(args)<5: continue
        name = args[1].split("_Tune")[0]
        while name.startswith("/"): name = name.split("/")[1]
        self.addDataset( Dataset( name, int(args[0]), float(args[2]), float(args[3]), float(args[4]) ) )


    def __init__(self, filename="samples_50ns_miniaod.txt", *args, **kwargs):
        self.update(*args, **kwargs)
        self.addDatasetsFromFile(filename)
