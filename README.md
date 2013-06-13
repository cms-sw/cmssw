GEMCode
======

Repository for GEM code development

Documentation
-------------

* Home page
https://twiki.cern.ch/twiki/bin/view/MPGD/GemSimulationsInstructionsCMSSW

* Geometry developments 
https://twiki.cern.ch/twiki/bin/view/MPGD/GEMGeometryRoadMap

* Digitizer
https://twiki.cern.ch/twiki/bin/view/MPGD/GEMDigitizationRoadMap

* Validation
https://twiki.cern.ch/twiki/bin/view/MPGD/GemSimulationsInstructionsCMSSW
http://cms-project-gem-validation.web.cern.ch/cms-project-gem-validation/

* Trigger
https://twiki.cern.ch/twiki/bin/view/MPGD/GEMTriggerRoadMap



------------------------------------
--- Instructions to get the code ---
------------------------------------

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

3. Project-specific setup
