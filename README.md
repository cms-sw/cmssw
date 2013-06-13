#GEMCode#

##Introduction##

This is the repository for code development of GEM the validation analyzer and the GEM-CSC integrated local trigger analyzer.<br>

This README.md file can is encoded in MarkDown, see<br>
http://daringfireball.net/projects/markdown/syntax

##Documentation##

* Home page of validation page<br>
https://twiki.cern.ch/twiki/bin/view/MPGD/GemSimulationsInstructionsCMSSW<br>

* Information on the geometry<br>
https://twiki.cern.ch/twiki/bin/view/MPGD/GEMGeometryRoadMap<br>

* Information on the digitizer developments<br>
https://twiki.cern.ch/twiki/bin/view/MPGD/GEMDigitizationRoadMap<br>

* Validation<br>
https://twiki.cern.ch/twiki/bin/view/MPGD/GemSimulationsInstructionsCMSSW<br>
http://cms-project-gem-validation.web.cern.ch/cms-project-gem-validation/<br>

* Road map of the development of the GEM-CSC integrated local trigger<br>
https://twiki.cern.ch/twiki/bin/view/MPGD/GEMTriggerRoadMap


##Instructions to get the code##

1. General GitHub setup

1.1 Get an account on GitHub. Follow the instructions<br>  
http://cms-sw.github.io/cmssw/faq.html#how_do_i_subscribe_to_github

1.2 Get an ssh key for each computer you would like to connect from (Lxplus, LPC,...)<br> 
<code>
ssh-keygen -t rsa -C "john_doe@spameggs.com"
</code>

Copy the content of id_rsa.pub to<br> 
https://github.com/settings/ssh. 

Test the validity of the key in your user area:<br>
<code>
ssh -T git@github.com
</code>

You should see a message:<br> 
Hi $USER_NAME! You've successfully authenticated, but GitHub does not provide shell access.<br>

1.3 Add to your bashrc file:<br> 
<code>
export CMSSW_GIT_REFERENCE=/afs/cern.ch/cms/git-cmssw-mirror/cmssw.git
</code>
See also the advanced FAQ: http://cms-sw.github.io/cmssw/advanced-usage

2. CMSSW-specific github setup

2.1 Setup a new CMSSW environment. See list of CMSSW tags on Git to get the latest version available (currently CMSSW_6_2_0_pre5).

<code>
cmsrel CMSSW_X_Y_Z<br>
cd CMSSW_X_Y_Z/src<br>
cmsenv<br>
</code>
 
2.2 Check out latest version of ReleaseScripts. You have to CMSSW_6_2_0_pre5, but is should be included in CMSSW_6_2_0_pre6 and CMSSW_6_2_0_pre7

<code>
cvs co Utilities/ReleaseScripts<br>
scram b -j 9<br>
</code>

2.3 Initialize and configure Git. In CMSSW_X_Y_Z/src, do

<code>
git init<br>
git config --list<br>
git config --global remote.cmssw-main.url git@github.com:cms-sw/cmssw.git<br>
git config --global remote.cmssw-main-ro.url https://github.com/cms-sw/cmssw.git<br>
git config --global core.sparsecheckout true<br>
</code>

If your user identity has not been set yet, do

<code>
git config --global core.editor emacs
git config --global user.name FirstName LastName
git config --global user.email FirstName.LastName@cern.ch
</code>

3. Project-specific setup







