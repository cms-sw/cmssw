
#
# \version $Id: MultiPhotonAnalyzer_cfi.py,v 1.2 2011/10/17 09:07:54 kimy Exp $
#

import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.PhotonAnalysis.SinglePhotonAnalyzer_cfi import *

multiPhotonAnalyzer = singlePhotonAnalyzer.clone()
multiPhotonAnalyzer._TypedParameterizable__type="MultiPhotonAnalyzerTree"



