create or replace view view_last_run as
SELECT 'Current run number' AS NAME,
       TO_CHAR ( MAX ( RUNNUMBER ) ) AS VALUE
  FROM FILES_INFO WHERE 
        RUNNUMBER = ( SELECT MAX ( RUNNUMBER ) FROM FILES_INFO 
                      WHERE  (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 )
 UNION SELECT 'Number of all files' AS NAME,
       TO_CHAR ( COUNT ( FILENAME ) ) AS VALUE
  FROM FILES_INFO WHERE 
        RUNNUMBER = ( SELECT MAX ( RUNNUMBER ) FROM FILES_INFO 
                      WHERE  (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 )
 UNION SELECT 'Number of closed files' AS NAME,
       TO_CHAR ( COUNT ( * ) ) AS VALUE
  FROM FILES_INFO WHERE 
        RUNNUMBER = ( SELECT MAX ( RUNNUMBER ) FROM FILES_INFO 
                      WHERE  (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 )
        and STATE>0
 UNION SELECT 'Number of copied files' AS NAME,
       TO_CHAR ( COUNT ( * ) ) AS VALUE
  FROM FILES_INFO WHERE 
        RUNNUMBER = ( SELECT MAX ( RUNNUMBER ) FROM FILES_INFO 
                      WHERE  (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 )
        and STATE>10
 UNION SELECT 'Number of hosts' AS NAME,
       TO_CHAR ( COUNT ( DISTINCT HOSTNAME ) ) AS VALUE
  FROM FILES_INFO WHERE 
        RUNNUMBER = ( SELECT MAX ( RUNNUMBER ) FROM FILES_INFO 
                      WHERE  (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 )
 UNION SELECT 'Number of streams' AS NAME,
       TO_CHAR ( COUNT ( DISTINCT STREAM ) ) AS VALUE
  FROM FILES_INFO WHERE 
        RUNNUMBER = ( SELECT MAX ( RUNNUMBER ) FROM FILES_INFO 
                      WHERE  (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 )
 UNION SELECT 'Number of luminosity sections' AS NAME,
       TO_CHAR ( COUNT ( DISTINCT LUMISECTION ) ) AS VALUE
  FROM FILES_INFO WHERE 
        RUNNUMBER = ( SELECT MAX ( RUNNUMBER ) FROM FILES_INFO 
                      WHERE  (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800 );
grant select on view_last_run to public;
