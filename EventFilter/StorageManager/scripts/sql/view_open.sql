create or replace view view_open as
SELECT 'Number of open files' AS DESCRIPTION,
       TO_CHAR ( COUNT ( * ) ) AS COUNT
  FROM FILES_INFO where state=0
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Oldest open file start time' AS DESCRIPTION,
       TO_CHAR ( MIN ( CREATED_TIME ) ,
		 'YYYY/MM/DD HH24:MI' ) AS COUNT
  FROM FILES_INFO where state=0
        and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
 UNION SELECT 'Oldest open file run number' AS DESCRIPTION,
       TO_CHAR ( RUNNUMBER ) AS COUNT
  FROM FILES_INFO where 
        CREATED_TIME = ( SELECT MIN ( CREATED_TIME )
                        FROM FILES_INFO where state=0 
                and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
        );
grant select on view_open to public;
