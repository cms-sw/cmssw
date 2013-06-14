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

Follow the instructions on<br>http://cms-sw.github.io/cmssw/faq.html#how_do_i_subscribe_to_github

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

2.1 Setup a new CMSSW environment. See list of CMSSW tags on Git to get the latest version available (currently CMSSW_6_2_0_pre5).
<pre><code>cmsrel CMSSW_X_Y_Z<br>cd CMSSW_X_Y_Z/src<br>cmsenv</code></pre>
 
2.2 Check out latest version of ReleaseScripts. 
You have to include it for CMSSW_6_2_0_pre5, but is should be included in CMSSW_6_2_0_pre6 and CMSSW_6_2_0_pre7.<pre><code>cvs co Utilities/ReleaseScripts<br>scram b -j 9</code></pre>

2.3 Initialize and configure Git. In CMSSW_X_Y_Z/src, do<pre><code>git init<br>git config --list<br>git config --global remote.cmssw-main.url git@github.com:cms-sw/cmssw.git<br>git config --global remote.cmssw-main-ro.url https://github.com/cms-sw/cmssw.git<br>git config --global core.sparsecheckout true</code></pre>

If your user identity has not been set yet, do<pre><code>git config --global core.editor emacs<br>git config --global user.name FirstName LastName<br>git config --global user.email FirstName.LastName@cern.ch</code></pre>


3. Project-specific setup

3.1 Get the private, customized CMSSW code

<pre><code>git remote add cmssw-gem git@github.com:gem-sw/cmssw.git<br>git fetch cmssw-gem<br>git pull cmssw-gem master<br></code></pre>

Check the available branches<pre><code>git branch</code></pre>

Check out your personal development branch<pre><code>git checkout -b mybranch/for/gemcode</code></pre>

3.2 Check out the packages you want to modify

<pre><code>git addpkg L1Trigger/CSCTriggerPrimitives<br>git addpkg L1Trigger/GlobalMuonTrigger<br>git addpkg L1Trigger/CSCTrackFinder<br>git addpkg L1Trigger/CSCCommonTrigger<br>git addpkg DataFormats/L1CSCTrackFinder<br>git addpkg DataFormats/CSCDigi<br>git addpkg DataFormats/GEMDigi<br>git addpkg DataFormats/GEMRecHit</code></pre>

Compile:<pre><code>scram b -j 9</code></pre>

3.3 Adding submodules

Validation code
<pre><code>git submodule add git://github.com/gem-sw/GEMCode.git</code></pre>

Trigger code
<pre><code>git submodule add git://github.com/gem-sw/SimMuL1.git</code></pre>

Website development
<pre><code>git submodule add git://github.com/gem-sw/Website.git</code></pre>

Check that you are on the master branch in each submodule. Create your personal branch for code development.

Compile<pre><code>scram b -j 9</code></pre>






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

2.1 
GitHub II}
\small
\begin{description}
\item[Step 4] Setup a new CMSSW environment (latest available on Git)
{\color{orange} \scriptsize
cmsrel CMSSW\_6\_2\_0\_pre5 \\
cd CMSSW\_6\_2\_0\_pre5/src \\
cmsenv
}
\item[Step 5] Check out utilities for Git which are not yet in CMSSW\_6\_2\_0\_pre5 \\ 
{\color{orange} \scriptsize
cvs co Utilities/ReleaseScripts \\
scram b -j 9}
\item[Step 6] Initialize Git in your environment\\
{\color{orange} \scriptsize
git init}
\item[Step 7] Check configuration \\
{\color{orange} \scriptsize
git config \texttt{--}list}
\item[Step 8] Set the location of the remote CMSSW repository \end{description}
{\scriptsize\color{orange}
git config \texttt{--}global remote.cmssw-main.url git@github.com:cms-sw/cmssw.git \\
git config \texttt{--}global remote.cmssw-main-ro.url https://github.com/cms-sw/cmssw.git \\
git config \texttt{--}global core.sparsecheckout true \\
}

3. Project-specific setup
