GEMCode
======

Introduction
-------------

This is the repository for code development of GEM the validation analyzer and the GEM-CSC integrated local trigger analyzer. 


Documentation
-------------

* Home page of validation page

https://twiki.cern.ch/twiki/bin/view/MPGD/GemSimulationsInstructionsCMSSW

* Information on the geometry

https://twiki.cern.ch/twiki/bin/view/MPGD/GEMGeometryRoadMap

* Information on the digitizer developments

https://twiki.cern.ch/twiki/bin/view/MPGD/GEMDigitizationRoadMap

* Validation

https://twiki.cern.ch/twiki/bin/view/MPGD/GemSimulationsInstructionsCMSSW

http://cms-project-gem-validation.web.cern.ch/cms-project-gem-validation/

* Road map of the development of the GEM-CSC integrated local trigger

https://twiki.cern.ch/twiki/bin/view/MPGD/GEMTriggerRoadMap


Instructions to get the code
----------------------------

1. General GitHub setup

1.1 Get an account on GitHub. Follow the instructions  
http://cms-sw.github.io/cmssw/faq.html#how_do_i_subscribe_to_github

1.2 Get an ssh key for each computer you would like to connect from (Lxplus, LPC,...) 
ssh-keygen -t rsa -C "john_doe@spameggs.com"

Copy the content of id_rsa.pub to https://github.com/settings/ssh. 
Test the validity of the key in your user area:ssh -T git@github.com
You should see a message: Hi $USER_NAME! You've successfully authenticated, but GitHub does not provide shell access.

1.3 Add to your bashrc file: 
export CMSSW_GIT_REFERENCE=/afs/cern.ch/cms/git-cmssw-mirror/cmssw.git
See also the advanced FAQ: http://cms-sw.github.io/cmssw/advanced-usage

2. CMSSW-specific github setup

2.1 Setup a new CMSSW environment. 

cmsrel CMSSW_X_Y_Z<br>
cd CMSSW_X_Y_Z/src<br>
cmsenv<br>
 
See list of CMSSW tags on Git to get the latest version available (currently CMSSW_6_2_0_pre5).


2.2 Check out latest version of ReleaseScripts

cvs co Utilities/ReleaseScripts

scram b -j 9

Must for CMSSW_6_2_0_pre5, included in CMSSW_6_2_0_pre6 and CMSSW_6_2_0_pre7


2.3 Initialize and configure Git and 

in CMSSW_X_Y_Z/src, do 

git init;
git config --list};
git config \texttt{--}global remote.cmssw-main.url git@github.com:cms-sw/cmssw.git \\
git config \texttt{--}global remote.cmssw-main-ro.url https://github.com/cms-sw/cmssw.git \\
git config \texttt{--}global core.sparsecheckout true \\


3. Project-specific setup

Note
----

This README.md file can is encoded in MarkDown. See also
http://daringfireball.net/projects/markdown/syntax

