create or replace view view_last as
SELECT '1: Latest start time' AS NAME,
       TO_CHAR ( MAX ( CREATED_TIME ),
		 'YYYY/MM/DD HH24:MI' ) AS VALUE
  FROM FILES_INFO where STATE>=0 and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
UNION SELECT '2: Latest close time' AS NAME,
       TO_CHAR ( MAX ( INJECTED_TIME ),
		 'YYYY/MM/DD HH24:MI' ) AS VALUE
  FROM FILES_INFO where STATE>=1 and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800
UNION SELECT '3: Latest transfer time' AS NAME,
       TO_CHAR ( MAX ( COPIED_TIME ),
		 'YYYY/MM/DD HH24:MI' ) AS VALUE
  FROM FILES_INFO where STATE>10  and (CAST( CREATED_TIME as DATE) - sysdate)*86400 > -604800;
grant select on view_last to public;
