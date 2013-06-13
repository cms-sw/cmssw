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

1.1 Get an account on GitHub. 

Follow the instructions on <br>  
http://cms-sw.github.io/cmssw/faq.html#how_do_i_subscribe_to_github

1.2 Get an ssh key for each computer you would like to connect from (Lxplus, LPC,...)
<pre>
<code>
ssh-keygen -t rsa -C "john_doe@spameggs.com"
</code>
</pre>

Copy the content of id_rsa.pub to<br> 
https://github.com/settings/ssh. 

Test the validity of the key in your user area:
<pre><code>
ssh -T git@github.com
</code></pre>

You should see a message:<br> 
Hi $USER_NAME! You've successfully authenticated, but GitHub does not provide shell access.

1.3 Add to your bashrc file:<br> 
<pre><code>
export CMSSW_GIT_REFERENCE=/afs/cern.ch/cms/git-cmssw-mirror/cmssw.git
</code></pre>

See also the advanced FAQ<br>
http://cms-sw.github.io/cmssw/advanced-usage<br>

2. CMSSW-specific github setup<br>

2.1 Setup a new CMSSW environment. See list of CMSSW tags on Git to get the latest version available (currently CMSSW_6_2_0_pre5).
<pre><code>
cmsrel CMSSW_X_Y_Z
cd CMSSW_X_Y_Z/src
cmsenv
</code></pre>
 
2.2 Check out latest version of ReleaseScripts. You have to CMSSW_6_2_0_pre5, but is should be included in CMSSW_6_2_0_pre6 and CMSSW_6_2_0_pre7
<pre><code>
cvs co Utilities/ReleaseScripts
scram b -j 9
</code></pre>

2.3 Initialize and configure Git. In CMSSW_X_Y_Z/src, do
<pre><code>
git init
git config --list
git config --global remote.cmssw-main.url git@github.com:cms-sw/cmssw.git
git config --global remote.cmssw-main-ro.url https://github.com/cms-sw/cmssw.git
git config --global core.sparsecheckout true
</code></pre>

If your user identity has not been set yet, do
<pre><code>
git config --global core.editor emacs
git config --global user.name FirstName LastName
git config --global user.email FirstName.LastName@cern.ch
</code></pre>

3. Project-specific setup

3.1 Get the private, customized CMSSW code

<pre><code>
git remote add cmssw-gem git@github.com:gem-sw/cmssw.git
git fetch cmssw-gemcsctrigger
git pull cmssw-gemcsctrigger gemcsctrigger
</code></pre>

Check the available branches
<pre><code>
git branch
</code></pre>

Check out your personal development branch
git checkout -b mybranch/for/gemcode

3.2 Check out the packages you want to modify

<pre><code>
git addpkg L1Trigger/CSCTriggerPrimitives
git addpkg L1Trigger/GlobalMuonTrigger                     
git addpkg L1Trigger/CSCTrackFinder
git addpkg L1Trigger/CSCCommonTrigger
git addpkg DataFormats/L1CSCTrackFinder
git addpkg DataFormats/CSCDigi
git addpkg DataFormats/GEMDigi
git addpkg DataFormats/GEMRecHit
</code></pre>

Compile 
<pre><code>
scram b -j 9
</code></pre>

3.3 Adding submodules

<pre>
Validation code
git submodule add git://github.com/GEMCSCTriggerDevelopers/GEMCSCTrigger.git
Trigger code
git submodule add git://github.com/GEMCSCTriggerDevelopers/GEMCSCTrigger.git
Website development
git submodule add git://github.com/GEMCSCTriggerDevelopers/GEMCSCTrigger.git
</pre>

Check that you are on the master branch in each submodule. Create your personal branch for code development.

Compile 
<pre><code>
scram b -j 9
</code></pre>










