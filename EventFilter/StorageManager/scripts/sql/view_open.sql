
create or replace view view_open as
SELECT 'Number of open files' AS DESCRIPTION,
       TO_CHAR ( COUNT ( * ) ) AS COUNT
  FROM FILES_CREATED 
 where not exists ( SELECT * from FILES_INJECTED where FILES_CREATED.FILENAME = FILES_INJECTED.FILENAME)
   AND PRODUCER='StorageManager'
 UNION SELECT 'Oldest open file Start Time' AS DESCRIPTION,
       TO_CHAR ( MIN ( CTIME ) ,
		 'YYYY/MM/DD HH24:MI' ) AS COUNT
  FROM FILES_CREATED 
 where not exists ( SELECT * from FILES_INJECTED where FILES_CREATED.FILENAME = FILES_INJECTED.FILENAME)
   AND PRODUCER='StorageManager'
 UNION SELECT 'Oldest open file Run Number' AS DESCRIPTION,
       TO_CHAR ( RUNNUMBER ) AS COUNT
  FROM FILES_CREATED 
  where CTIME = ( SELECT MIN ( CTIME )
                   FROM FILES_CREATED 
		   where not exists ( SELECT * from FILES_INJECTED where FILES_CREATED.FILENAME = FILES_INJECTED.FILENAME) 
                      AND PRODUCER='StorageManager' )
   AND PRODUCER='StorageManager';

grant select on view_open to public;
