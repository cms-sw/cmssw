explain plan set statement_id='allfillschemepattern' for select fillschemepattern,correctionfactor from cms_lumi_prod.fillscheme;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='allfillschemepattern' order by id;

explain plan set statement_id='fillscheme' for select fillnum,runnum,fillscheme,ncollidingbunches from cms_lumi_prod.cmsrunsummary where amodetag='PROTPHYS' and egev>=3500; 

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='fillscheme' order by id;

explain plan set statement_id='cmsrunsummary' for select l1key,amodetag,egev,hltkey,fillnum,sequence,to_char("STARTTIME",'MM/DD/YY HH24:MI:SS.FF6'), to_char("STOPTIME",'MM/DD/YY HH24:MI:SS.FF6') FROM cms_lumi_prod.cmsrunsummary where runnum=173380;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='cmsrunsummary' order by id;

explain plan set statement_id='lumidataid' for select data_id from cms_lumi_prod.lumidata where runnum=173380;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='lumidataid' order by id;

explain plan set statement_id='trgdataid' for select data_id from cms_lumi_prod.trgdata where runnum=173380;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='trgdataid' order by id;

explain plan set statement_id='instlumi' for select runnum,lumilsnum,cmslsnum,instlumi,beamstatus,beamenergy,numorbit,startorbit from cms_lumi_prod.lumisummaryv2 where data_id=564;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='instlumi' order by id;

explain plan set statement_id='deadtime' for select runnum,cmslsnum,deadtimecount,bitzerocount,bitzeroprescale,deadfrac from cms_lumi_prod.lstrg where data_id=546;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='deadtime' order by id;

explain plan set statement_id='normid' for select data_id from cms_lumi_prod.luminorms where amodetag='PROTPHYS' and egev_1>=3100 and egev_1<=3600;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='normid' order by id;

explain plan set statement_id='luminorms' for select entry_name,amodetag,norm_1,egev_1,norm_2,egev_2 from cms_lumi_prod.luminorms where data_id=1;

select cardinality "rows",lpad(' ',level-1)||operation||' '||options||' '||object_name "plan" from plan_table connect by prior id=parent_id and prior statement_id=statement_id start with id=0 and statement_id='luminorms' order by id;