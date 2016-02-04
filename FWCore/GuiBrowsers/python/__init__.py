#Automatically created by SCRAM
import os
localrt=os.getenv('LOCALRT', None)
arch=os.getenv('SCRAM_ARCH', None)
if localrt != None:
  __path__.append(localrt+'/cfipython/'+arch+'/FWCore/GuiBrowsers')
