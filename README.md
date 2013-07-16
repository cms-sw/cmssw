#GEMCode#

##Introduction##

This is the repository for code development of GEM the validation analyzer and the GEM-CSC integrated local trigger analyzer.<br><br>
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

Follow the instructions on<br>
http://cms-sw.github.io/cmssw/index.html

1.2 Get an ssh key for each computer you would like to connect from (Lxplus, LPC,...)<pre><code>ssh-keygen -t rsa -C "john_doe@spameggs.com"</code></pre>

Copy the content of id_rsa.pub to<br> 
https://github.com/settings/ssh. 

Test the validity of the key in your user area:<pre><code>ssh -T git@github.com</code></pre>

You should see a message:<br> 
Hi $USER_NAME! You've successfully authenticated, but GitHub does not provide shell access.

1.3 Add to your bashrc file:<br><pre><code>export CMSSW_GIT_REFERENCE=/afs/cern.ch/cms/git-cmssw-mirror/cmssw.git</code></pre>

See also the advanced FAQ<br>
http://cms-sw.github.io/cmssw/advanced-usage

1.4 Become a member of gem-sw. Send an email with your Git username to <br>
sven.dildick@cern.ch

2. CMSSW-specific github setup<br>

2.1 Setup a new CMSSW environment. See list of CMSSW tags on Git to get the latest version available (currently CMSSW_6_2_0).
<pre><code>cmsrel CMSSW_X_Y_Z<br>cd CMSSW_X_Y_Z/src<br>cmsenv</code></pre>
 
2.2 Initialize and configure Git. In CMSSW_X_Y_Z/src, do

<pre><code>git init
git config --list
git config --global remote.official-cmssw.url git@github.com:cms-sw/cmssw.git
git config --global remote.gem-cmssw.url git@github.com:gem-sw/cmssw.git
git config --global core.sparsecheckout true
</code></pre>

If your user identity has not been set yet, do<pre><code>git config --global core.editor emacs<br>git config --global user.name FirstName LastName<br>git config --global user.email FirstName.LastName@cern.ch</code></pre>


3. Project-specific setup

3.1 Get the private, customized CMSSW code

<pre><code>git cms-addpkg Geometry/GEMGeometry
git cms-addpkg Geometry/GEMGeometryBuilder
git cms-addpkg DataFormats/MuonDetId
git cms-addpkg DataFormats/GEMRecHit
git cms-addpkg DataFormats/CSCDigi
git cms-addpkg L1Trigger/CSCTriggerPrimitives
git cms-addpkg L1Trigger/GlobalMuonTrigger
git cms-addpkg DataFormats/L1CSCTrackFinder
git cms-addpkg L1Trigger/CSCTrackFinder
git cms-addpkg L1Trigger/CSCCommonTrigger
git cms-addpkg SimMuon/GEMDigitizer
</code></pre>

The current branch should be from-CMSSW_6_2_0. 

3.2 Checkout the latest GEM developments, merge the changes and compile.

<pre><code>git fetch cmssw-gem
git merge FETCH_HEAD
scram b -j 9
</code></pre>

3.3 Adding submodules

Validation code
<pre><code>git submodule add git://github.com/gem-sw/GEMCode.git</code></pre>

L1TriggerDevelopment:
<pre><code>git submodule add git://github.com/gem-sw/L1TriggerDPGUpgrade.git</code></pre>

Website development
<pre><code>git submodule add git://github.com/gem-sw/Website.git</code></pre>

Check that you are on the master branch in each submodule. Create your personal branch for code development.

Compile<pre><code>scram b -j 9</code></pre>










