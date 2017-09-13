#will remove this once ROOT random is set up and working in cpp
from ROOT import gSystem
gSystem.Load("libpapascpp") #check with Colin if this is OK or if should be made to execute just once
from ROOT import  randomgen

def expovariate (a):
    return randomgen.RandExponential(a).next()

def uniform (a, b):
    return randomgen.RandUniform(a, b).next()

def gauss (a, b):
    return randomgen.RandNormal(a, b).next()

def seed (s):
    randomgen.RandUniform(0, 1).setSeed(s)