=== Compile Instructions ===

cmsrel CMSSW_14_2_0_pre3
cd CMSSW_14_2_0_pre3/src
cmsenv
git cms-checkout-topic -u tomalin:masterP2TrackerUnpackers

=== Rebase Instructions ===

If you have a personal branch of this code, and wish to update it with changes made by others to tomalin:masterP2TrackerUnpackers , then in new project area:

git cms-checkout-topic -u  tomalin:masterP2TrackerUnpackers
git cms-rebase-topic -u myFork:myBranch
