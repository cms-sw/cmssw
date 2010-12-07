original: 

select l.PATHNAME,l.LSNUMBER,m.PSINDEX,m.PSVALUE,l.L1PASS,l.PACCEPT from cms_runinfo.hlt_supervisor_lumisections_v2 l, cms_runinfo.hlt_supervisor_scalar_map m where l.RUNNR=m.RUNNR and l.PSINDEX=m.PSINDEX and l.PATHNAME=m.PATHNAME and l.RUNNR=152658 order by l.LSNUMBER

hack:
hack with comparison
select distinct ld.PATHNAME,ld.LSNUMBER,ld.PSINDEX,li.LSNUMBER,li.PSINDEX,m.PSVALUE,ld.L1PASS,ld.PACCEPT from cms_runinfo.hlt_supervisor_lumisections_v2 ld,cms_runinfo.hlt_supervisor_lumisections_v2 li,cms_runinfo.hlt_supervisor_scalar_map m where ld.LSNUMBER=li.LSNUMBER+1 and ld.RUNNR=li.RUNNR and li.PSINDEX=m.PSINDEX and li.PATHNAME=m.PATHNAME and ld.PATHNAME=li.PATHNAME and li.RUNNR=152658 and m.PATHNAME='HLT_NewPrescale1' order by ld.LSNUMBER;

inuse:
#this is special only for LS=1
select l.PATHNAME,l.LSNUMBER,m.PSVALUE,l.L1PASS,l.PACCEPT from  cms_runinfo.hlt_supervisor_lumisections_v2 l,cms_runinfo.hlt_supervisor_scalar_map m where l.RUNNR=m.RUNNR and l.PSINDEX=m.PSINDEX and l.PATHNAME=m.PATHNAME and l.RUNNR=152658 and l.LSNUMBER=1; 
 
#this is the hack for the rest of the LS where self-join to find the psindex of the previous LS, therefore the first one is special.

select distinct ld.PATHNAME,ld.LSNUMBER,m.PSVALUE,ld.L1PASS,ld.PACCEPT from cms_runinfo.hlt_supervisor_lumisections_v2 ld,cms_runinfo.hlt_supervisor_lumisections_v2 li,cms_runinfo.hlt_supervisor_scalar_map m where ld.LSNUMBER=li.LSNUMBER+1 and ld.RUNNR=li.RUNNR and li.PSINDEX=m.PSINDEX and li.PATHNAME=m.PATHNAME and ld.PATHNAME=li.PATHNAME and li.RUNNR=152658 and m.PATHNAME='HLT_NewPrescale1' order by ld.LSNUMBER; 
