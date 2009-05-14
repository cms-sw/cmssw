create or replace view view_closed as
SELECT 'Number of closed files' AS DESCRIPTION,
       TO_CHAR ( COUNT ( * ) ) AS COUNT
  FROM FILES_INFO where state>0 
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Number of files (safety > 0)' AS DESCRIPTION,
       TO_CHAR ( COUNT ( * ) ) AS COUNT
  FROM FILES_INFO where state>1 
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Number of files (safety > 99)' AS DESCRIPTION,
       TO_CHAR ( COUNT ( * ) ) AS COUNT
  FROM FILES_INFO where state>10 
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Total size of closed files' AS DESCRIPTION,
       TO_CHAR ( ROUND ( SUM ( FILESIZE ) / 1073741824,
			 2 ) ) || ' GB' AS COUNT
  FROM FILES_INFO where state>0 
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Total size of files (safety > 0)' AS DESCRIPTION,
       TO_CHAR ( ROUND ( SUM ( FILESIZE ) / 1073741824,
			 2 ) ) || ' GB' AS COUNT
  FROM FILES_INFO where state>1 
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Total size of files (safety > 99)' AS DESCRIPTION,
       TO_CHAR ( ROUND ( SUM ( FILESIZE ) / 1073741824,
			 2 ) ) || ' GB' AS COUNT
  FROM FILES_INFO where state>10
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Oldest closed file run number' AS DESCRIPTION,
       TO_CHAR ( RUNNUMBER ) AS COUNT
  FROM FILES_INFO
 WHERE CREATED_TIME = ( SELECT MIN ( CREATED_TIME )
		        FROM FILES_INFO where state>0
                        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 )
UNION SELECT 'Oldest closed file start time' AS DESCRIPTION,
       TO_CHAR ( MIN (CREATED_TIME) , 'YYYY/MM/DD HH24:MI' ) AS COUNT
  FROM FILES_INFO
 WHERE CREATED_TIME = ( SELECT MIN ( CREATED_TIME )
		        FROM FILES_INFO where state>0
                        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 );
grant select on view_closed to public;
