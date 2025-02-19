/*
 * This script will get info about tables containin some IOV
 *
 * It creates a script that contains the actual commands to get
 * the maximum IOV value and the size (in GB) of each table 
 *
 * Usage:
 * 1. Run this script with the command @findIovfields
 * 
 * Copyright (C) by Giovanni.Organtini@roma1.infn.it 2009
 *
 */

/* basic steup */
SET NEWPAGE 0
SET SPACE 0
SET LINESIZE 80
SET PAGESIZE 0
SET ECHO OFF
SET FEEDBACK OFF
SET HEADING OFF
SET MARKUP HTML OFF

/* create a script */
SPOOL GETINFO.sql

/* the script contains lines in the format SELECT table_name, MAX(iov),
   table_size FROM table_name */ 
select 'select ''', tname, ''', to_char(max(', cname, '), ''9999999999''), ', 
  'TO_CHAR(', tsize/1024/1024/1024 , ', ''999.99'')', 
  ' from ', tname, ';' from (
  select t.table_name tname, t.column_name cname, s.bytes tsize, 
    t.data_type from 
    user_tab_cols t, user_all_tables a,
    user_segments s 
    where t.table_name = a.table_name and (
          t.column_name like '%IOV%' and t.data_type like '%NUMBER%') and
          t.table_name = s.segment_name
);

SPOOL OFF

/* run the script */
@GETINFO

