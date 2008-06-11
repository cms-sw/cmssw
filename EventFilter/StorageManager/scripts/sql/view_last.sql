create or replace view view_last as
SELECT 'Latest Open Time' AS NAME,
       TO_CHAR ( MAX ( CTIME ) , 'YYYY/MM/DD HH24:MI' ) AS VALUE
  FROM FILES_CREATED
  
 UNION SELECT 'Latest Close Time' AS NAME,
       TO_CHAR ( MAX ( ITIME ),
		 'YYYY/MM/DD HH24:MI' ) AS VALUE
  FROM FILES_INJECTED

 UNION SELECT 'Latest Trans Time' AS NAME,
       TO_CHAR ( MAX ( ITIME ),
		 'YYYY/MM/DD HH24:MI' ) AS VALUE
  FROM FILES_TRANS_CHECKED;

grant select on view_last to public;
