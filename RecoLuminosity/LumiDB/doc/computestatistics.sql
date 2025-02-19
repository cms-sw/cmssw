exec dbms_stats.gather_table_stats('','TRGDATA',no_invalidate=>false);
exec dbms_stats.gather_table_stats('','LSTRG',no_invalidate=>false);
exec dbms_stats.gather_table_stats('','LUMISUMMARYV2',no_invalidate=>false);
exec dbms_stats.gather_table_stats('','CMSRUNSUMMARY',no_invalidate=>false);
exec dbms_stats.gather_table_stats('','HLTDATA',no_invalidate=>false);
exec dbms_stats.gather_table_stats('','LSHLT',no_invalidate=>false);