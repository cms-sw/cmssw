update lumidata set "NCOLLIDINGBUNCHES"=(select "NCOLLIDINGBUNCHES" from cmsrunsummary where cmsrunsummary.runnum=lumidata.runnum) ;
