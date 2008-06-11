create or replace view view_last_five_files as
SELECT "FILE_NAME",
       "START_TIME"
  FROM ( SELECT FILENAME AS FILE_NAME,
		TO_CHAR ( CTIME , 'YYYY/MM/DD HH24:MI' ) AS START_TIME
	  FROM FILES_CREATED
          WHERE PRODUCER='StorageManager'
	  ORDER BY START_TIME DESC )
 WHERE ROWNUM < 6;

grant select on view_last_five_files to public;
