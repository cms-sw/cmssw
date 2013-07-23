[TOC]

GEMCode
=======

Introduction
------------

This is the repository for code development of GEM the validation analyzer and the GEM-CSC integrated local trigger analyzer.<br><br>
This README.md file can is encoded in MarkDown, see<br>
http://daringfireball.net/projects/markdown/syntax

Documentation
-------------

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

###General GitHub setup

Follow the instructions on<br>
http://cms-sw.github.io/cmssw/index.html

###CMSSW environment setup

<pre><code>cmsrel CMSSW_6_1_2_SLHC6_patch1
cd CMSSW_6_1_2_SLHC6_patch1/src
cmsenv
git init
</code></pre>

###Get the official CMSSW code

<pre><code>git cms-addpkg Geometry/GEMGeometry
git cms-addpkg Geometry/GEMGeometryBuilder
git cms-addpkg DataFormats/MuonDetId
git cms-addpkg DataFormats/GEMRecHit
git cms-addpkg DataFormats/CSCDigi
git cms-addpkg L1Trigger/CSCTriggerPrimitives
git cms-addpkg L1Trigger/GlobalMuonTrigger
git cms-addpkg DataFormats/L1CSCTrackFinder
git cms-addpkg SimMuon/GEMDigitizer
</code></pre>

###Checking out and merging the latest GEM developments

<!--
<pre><code>git cms-merge-topic dildick:bugfix-for-gemgeometry
git cms-merge-topic dildick:feature-for-gemgeometry
git cms-merge-topic dildick:feature-for-gemcsctrigger
</code></pre>
-->
<pre><code>git fetch cmssw-gem
git merge FETCH_HEAD
</code></pre>

Compile:<pre><code>scram b -j 9</code></pre>

###Adding submodules

Validation code
<pre><code>git submodule add git://github.com/gem-sw/GEMCode.git</code></pre>

L1TriggerDevelopment:
<pre><code>git submodule add git://github.com/gem-sw/L1TriggerDPGUpgrade.git</code></pre>

Website development
<pre><code>git submodule add git://github.com/gem-sw/Website.git</code></pre>

Check that you are on the master branch in each submodule. Create a new branch for each development.

Compile<pre><code>scram b -j 9</code></pre>
