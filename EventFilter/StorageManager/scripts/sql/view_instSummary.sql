CREATE OR REPLACE FUNCTION INSTANCE_CHECK(run in number, maxInstance in number, currentSETUP in VARCHAR2,  numInstance in number, maxClosed in number)
    RETURN NUMBER AS
    status NUMBER(5);
    lastNumInstance NUMBER(5);
    nInstance NUMBER(5);
    nCount NUMBER(5);
BEGIN
    status := 0;

    --Won't perform some checks if no instance has at least 25 closed files 
    IF (maxClosed > 25) THEN

    --Retrieves the number of active instances from the preceding run with the same setup label
    SELECT COUNT(N_CREATED) INTO lastNumInstance FROM (SELECT RUNNUMBER, N_CREATED, SETUPLABEL, DENSE_RANK() OVER (ORDER BY RUNNUMBER DESC NULLS LAST) RUN_RANK FROM (SELECT * FROM SM_INSTANCES WHERE RUNNUMBER < run AND SETUPLABEL = currentSETUP)) WHERE RUN_RANK = 1;


    --If the current run has a different number of active instances than the preceding run of the same setup label turn magenta
    IF (numInstance != lastNumInstance) THEN
         status := 1;
    END IF;
    
--    --If the number of active instances does not match the maximum observed instance (ie there is a gap) turn magenta
--    IF (numInstance != maxInstance + 1) THEN
--         status := 1;
--    END IF;


END IF;


    --If the number instances is ZERO, then something is wrong!
    IF (numInstance = 0) THEN
         status := 2;
    END IF;



    --compare number of non-zero files created with number of instances:
    SELECT COUNT(INSTANCE), COUNT(N_CREATED) into nInstance, nCount  FROM SM_INSTANCES WHERE RUNNUMBER=run;
    IF ( nInstance - nCount > 0) THEN
         status := 2;
    END IF;

     
    RETURN status;
END INSTANCE_CHECK; 
/ 


CREATE OR REPLACE FUNCTION NOTFILES_CHECK(run in number, maxUnacc in number, maxClosed in number,  maxLastWrite in DATE)  
    RETURN NUMBER AS
    status NUMBER(5);
    lastNumInstance NUMBER(5);
    nInstance NUMBER(5);
    nCount NUMBER(5);
    nStream NUMBER(5);
BEGIN
    status := 0;


   --number of Unacc files during running can be only a few times number of streams:
   SELECT COUNT(CMS_STOMGR.SM_SUMMARY.STREAM)  INTO nStream from CMS_STOMGR.SM_SUMMARY WHERE  RUNNUMBER = run;
 
	IF (   (time_diff(sysdate, maxLastWrite) > 30*60 AND maxUnacc > 0 ) OR
               (time_diff(sysdate, maxLastWrite) < 30*60 AND maxUnacc > 3*nStream+5  )  )  THEN
	  status := 1;

	END IF;
 

	IF (   (time_diff(sysdate, maxLastWrite) > 45*60 AND maxUnacc > 0 ) OR
               (time_diff(sysdate, maxLastWrite) < 45*60 AND maxUnacc > 5*nStream+5  )    )  THEN
	  status := 2;

	END IF;
 

    RETURN status;
END  NOTFILES_CHECK; 
/ 





CREATE OR REPLACE FUNCTION GENERATE_FLAG_NOTFILES(run in number, maxUnacc in number,  maxClosed in number, maxLastWrite in DATE)
    RETURN VARCHAR2 AS
    flag VARCHAR2(1000);
    threshold NUMBER(10);
    numFlagged NUMBER(5); 
    nstream NUMBER(10);
BEGIN
    --Determine minimum below which should be flagged.

    flag := ' ';
    numFlagged := 0;
    --IF (maxUnacc = 0) THEN
        -- RETURN flag;
    --END IF;


    SELECT COUNT(CMS_STOMGR.SM_SUMMARY.STREAM)  INTO nstream from CMS_STOMGR.SM_SUMMARY WHERE  RUNNUMBER = run;


    --Loop through all of the instances for this run
    FOR entry IN (SELECT * FROM SM_INSTANCES WHERE RUNNUMBER = run ORDER BY INSTANCE ASC)
    LOOP
	--Check if the difference between the last write on this instance and the most recent write on any instance is less than 30 minutes, on that basis, decide what is acceptable file count


	IF (   (time_diff(sysdate,  entry.Last_Write_Time) > 30*60 AND entry.N_UNACCOUNT > 0 ) OR
               (time_diff(sysdate,  entry.Last_Write_Time) < 30*60 AND entry.N_UNACCOUNT > 3*nStream+5 AND maxClosed > 25 )  )  THEN
 
		numFlagged := numFlagged + 1;
		IF (numFlagged = 1) THEN
			flag := flag || '!File:';
		END IF;
		flag := flag || ' ' || TO_CHAR(entry.INSTANCE) || '(' || TO_CHAR(NVL(entry.N_UNACCOUNT,0)) || ')';  --Flag this instance
	END IF;
    END LOOP;


    RETURN flag;
END GENERATE_FLAG_NOTFILES;
/


CREATE OR REPLACE FUNCTION GENERATE_FLAG_CLOSED(run in number, maxNum in number, maxLastWrite in DATE)
    RETURN VARCHAR2 AS
    flag VARCHAR2(1000);
    threshold NUMBER(10);
    numFlagged NUMBER(5); 
BEGIN
    --Determine minimum below which should be flagged.

    flag := ' ';
    numFlagged := 0;

   --Won't perform check if no instance has at least 25 closed files 
    IF (maxNum < 25) THEN
         RETURN flag;
    END IF;


    --Loop through all of the instances for this run
    FOR entry IN (SELECT * FROM SM_INSTANCES WHERE RUNNUMBER = run ORDER BY INSTANCE ASC)
    LOOP
	--Check if the difference between the last write on this instance and the most recent write on any instance is greater than 5 minutes
	IF ( (time_diff(maxLastWrite, entry.Last_Write_Time) > 660) OR 
             (maxNum - NVL(entry.N_INJECTED, 0) > .33 * maxNum AND maxNum>50 ))  THEN
		numFlagged := numFlagged + 1;
		IF (numFlagged = 1) THEN
			flag := flag || 'CLOSED:';
		END IF;
		flag := flag || ' ' || TO_CHAR(entry.INSTANCE) || '(' || TO_CHAR(NVL(entry.N_INJECTED,0)) || ')';  --Flag this instance
	END IF;
    END LOOP;
  
    RETURN flag;
END GENERATE_FLAG_CLOSED;
/

CREATE OR REPLACE FUNCTION GENERATE_FLAG_INJECTED(run in number, maxRatio in number, sumNum in number)
    RETURN VARCHAR2 AS
    flag VARCHAR2(1000);
    threshold NUMBER(10);
    numFlagged NUMBER(5); 
BEGIN
    flag := ' ';
    numFlagged := 0;


    --Loop through all instances for this run
    FOR entry IN (SELECT * FROM SM_INSTANCES WHERE RUNNUMBER = run ORDER BY INSTANCE ASC)
    LOOP
	--Formula to check if this instance is lagging the max instance by too much or is not injecting
	IF ( ( 0.7 * NVL(entry.N_INJECTED,0) * maxRatio > NVL(entry.N_NEW,0) AND entry.N_INJECTED > 20) OR (sumNum > 50 AND NVL(entry.N_NEW,0) = 0)) THEN
		numFlagged := numFlagged + 1;
		IF (numFlagged = 1) THEN
			flag := flag || 'INJECTED:';
		END IF;
		flag := flag || ' ' || TO_CHAR(entry.INSTANCE) || '(' || TO_CHAR(NVL(entry.N_NEW,0)) || ')'; --flag the instance
	END IF;
    END LOOP;

    RETURN flag;
END GENERATE_FLAG_INJECTED;
/

CREATE OR REPLACE FUNCTION GENERATE_FLAG_TRANSFERRED(run in number, maxRatio in number, sumNum in number)
    RETURN VARCHAR2 AS
    flag VARCHAR2(1000);
    threshold NUMBER(10);
    numFlagged NUMBER(5); 
BEGIN
    flag := ' ';
    numFlagged := 0;
    --Loop through all the instances for this run
    FOR entry IN (SELECT * FROM SM_INSTANCES WHERE RUNNUMBER = run ORDER BY INSTANCE ASC)
    LOOP
	--Formula to check if this instance is lagging the max instance by too much or not transferring
	IF ( (sumNum > 75 AND  0.7 * NVL(entry.N_NEW,0) * maxRatio > NVL(entry.N_COPIED,0) ) OR 
             (sumNum > 50 AND NVL(entry.N_COPIED,0) = 0)) THEN
		numFlagged := numFlagged + 1;
		IF (numFlagged = 1) THEN
			flag := flag || 'TRANS:';
		END IF;
		flag := flag || ' ' || TO_CHAR(entry.INSTANCE) || '(' || TO_CHAR(NVL(entry.N_COPIED,0)) || ')'; --flag the instance
	END IF;
    END LOOP;

    RETURN flag;
END GENERATE_FLAG_TRANSFERRED;
/

CREATE OR REPLACE FUNCTION GENERATE_FLAG_CHECKED(run in number, maxRatio in number, sumNum in number)
    RETURN VARCHAR2 AS
    flag VARCHAR2(1000);
    threshold NUMBER(10);
    numFlagged NUMBER(5); 
BEGIN
    flag := ' ';
    numFlagged := 0;
    --Loop through all the instances for this run
    FOR entry IN (SELECT * FROM SM_INSTANCES WHERE RUNNUMBER = run ORDER BY INSTANCE ASC)
    LOOP
	--Formula to check if this instance is lagging the max instance by too much or not checking
	IF ( (sumNum > 75 AND 0.7 * NVL(entry.N_NEW,0) * maxRatio > NVL(entry.N_CHECKED,0) ) OR 
             (sumNum > 50 AND NVL(entry.N_CHECKED,0) = 0)) THEN
		numFlagged := numFlagged + 1;
		IF (numFlagged = 1) THEN
			flag := flag || 'CHECKED:';
		END IF;
		flag := flag || ' ' || TO_CHAR(entry.INSTANCE) || '(' || TO_CHAR(NVL(entry.N_CHECKED,0)) || ')'; --flag the instance
	END IF;
    END LOOP;

    RETURN flag;
END GENERATE_FLAG_CHECKED;
/

CREATE OR REPLACE FUNCTION GENERATE_FLAG_DELETED(run in number, StartRunTime in DATE, maxLastClosedTime in DATE)
    RETURN VARCHAR2 AS
    flag VARCHAR2(1000);
    numFlagged NUMBER(5); 
    slength    NUMBER(5);
    rack       NUMBER(5);
    node       NUMBER(5);
    hour6      NUMBER(5);
    mins       NUMBER(10);
    hourPC     NUMBER(10);
    minPC      NUMBER(10);
    hourPC3    NUMBER(10);
    minPC20    NUMBER(10);
    deltaT     NUMBER(10,2);
    tLastDelete  DATE;
    tdeletable NUMBER(10,2);
    trunning   NUMBER(10,2);
    nexpfiles  NUMBER(10,2);
BEGIN

    flag := ' ';
    numFlagged := 0;


    --hour of the day:
    hour6   := time_diff( TRUNC(sysdate,'HH24'), TRUNC(sysdate,'DDD') )/3600;
    --min of the hour:
    mins    := time_diff( TRUNC(sysdate,'MI'),   TRUNC(sysdate,'HH24') )/60;
    hour6   := mod(hour6, 6);

--DBMS_OUTPUT.PUT_LINE (TO_CHAR(TRUNC(sysdate,'HH24'), 'Dy Mon DD HH24:MI:SS YYYY') || '  ' || TO_CHAR(TRUNC(sysdate,'DDD'), 'Dy Mon DD HH24:MI:SS YYYY')|| ' ' || TO_CHAR(TRUNC(sysdate,'MI'), 'Dy Mon DD HH24:MI:SS YYYY')  );


--DBMS_OUTPUT.PUT_LINE ('  hour6  mins : ' ||  hour6 || ' ' ||  mins ||   '   << ' );


    --Loop through all the instances
    FOR entry IN (SELECT * FROM SM_INSTANCES WHERE RUNNUMBER = run ORDER BY INSTANCE ASC)
    LOOP

--DBMS_OUTPUT.PUT_LINE (run  || '  ' ||  entry.HOSTNAME || ' Enter LOOP....   << ' );

	IF (  NVL(entry.N_CHECKED,0) > NVL(entry.N_DELETED,0) ) THEN
	        slength := length(entry.HOSTNAME);
		rack    := to_number(substr(entry.HOSTNAME,slength-4,2));
        	node    := to_number(substr(entry.HOSTNAME,slength-1,2));


--DBMS_OUTPUT.PUT_LINE ( entry.HOSTNAME || '  ' || rack  || '  ' || node || ' are rack  node   << ' );


	        IF  ( rack = 5 ) THEN      --minidaq! [maybe more general to just be NOT rack 6,7??
			--key: 60 sec/min * ( 60 min/hr * (15:00 minidaq delete time + 2 hr delete delay) + 20 min grace)
			IF ( time_diff(TRUNC(sysdate,'DD'), maxLastClosedTime) - 60*(60*(15+2) + 20) > 0   ) THEN
				numFlagged := numFlagged + 1;
				IF (numFlagged = 1) THEN
					flag := flag || 'DELETED:';
				END IF;
				flag := flag || ' ' || TO_CHAR(entry.INSTANCE) || '(' || TO_CHAR(NVL(entry.N_DELETED,0)) || ')';  --flag the instance 	
		         END IF; 
	        ELSIF ( rack = 6 OR rack = 7 ) THEN
			hourPC  := (rack-6) + mod(2*(node-12) + 10*(rack-6), 18);
  			minPC   := mod(hourPC,3);
          		hourPC3 := (hourPC - minPC)/3;
          		minPC20 := 20*minPC;

          		--time diff relative to now for scheduled deletes for this node--in MIN 
                        --  (generous 25 min offset to keep ahead of any variance within a PC's window)::
          		deltaT  :=  (hour6 - hourPC3) +  (mins - minPC20 - 25)/60.0;  


 --DBMS_OUTPUT.PUT_LINE ( ' deltaT= '|| TO_CHAR(deltaT,'9999.99')  || ' hourPC= '  ||  hourPC  ||  ' minPC= '  ||  minPC  ||  '  hourPC3= '  ||   hourPC3  ||  ' minPC20= '  || minPC20  ||   '   << ' );
  

                        IF( deltaT <  0) THEN
				deltaT := 6 + deltaT;
                        END IF;

                        --estimated sysdate of last delete for this node:
                        tLastDelete := sysdate - deltaT/24;
--DBMS_OUTPUT.PUT_LINE ( 'estimated last delete time: ' || TO_CHAR(tLastDelete, 'DD HH24:MI:SS')  );


                        tdeletable := time_diff(tLastDelete,StartRunTime) - 3.0*3600;
                        if ( tdeletable < 0 ) THEN
				tdeletable := 0;
			END IF;
                        trunning   := time_diff( maxLastClosedTime, StartRunTime)+23;  


--DBMS_OUTPUT.PUT_LINE ( 'tLastDelete= ' || tLastDelete || ' tdeletable= ' || tdeletable|| '  trunning= ' ||  trunning  ||  '   << ' );

                        nexpfiles := NVL(entry.N_CHECKED,0)*tdeletable/trunning;
			IF( tdeletable < 0 ) THEN
				 nexpfiles := 0;
                        ELSIF ( tdeletable/trunning > 1 ) THEN
				 nexpfiles := NVL(entry.N_CHECKED,0);
                        END IF;


	
--DBMS_OUTPUT.PUT_LINE ( 'nexpfiles= ' || nexpfiles ||  ' ?? entry.N_DELETED= ' || TO_CHAR(NVL(entry.N_DELETED,0)) || '  << ' );



			IF ( NVL(entry.N_DELETED,0) < nexpfiles   ) THEN
				numFlagged := numFlagged + 1;
				IF (numFlagged = 1) THEN
					flag := flag || 'DELETED:';
				END IF;
				flag := flag || ' ' || TO_CHAR(entry.INSTANCE) || '(' || TO_CHAR(NVL(entry.N_DELETED,0)) || ')';  --flag the instance 				END IF;
	
		        END IF; 

		END IF;

	END IF;

    END LOOP;

    RETURN flag;
END GENERATE_FLAG_DELETED;
/

--Provides per run information about the sm instances (one row per run) including the min and max counts for each stage and flagged suspicious instances
create or replace view view_sm_instance_summary
AS SELECT "RUN_NUMBER",
          "N_INSTANCES",
          "MINMAX_NOTFILES",
          "MINMAX_CLOSED",
	  "MINMAX_INJECTED",
          "MINMAX_TRANSFERRED",
          "MINMAX_CHECKED",
          "MINMAX_DELETED",
          "INSTANCE_STATUS",
          "NOTFILES_STATUS",
          "FLAGS",
          "RANK"
FROM (SELECT TO_CHAR( RUNNUMBER ) AS RUN_NUMBER,
             TO_CHAR( TO_CHAR(COUNT(N_CREATED)) || '/' || TO_CHAR(MAX(INSTANCE) + 1)  ) AS N_INSTANCES,
             TO_CHAR( TO_CHAR(MIN(NVL(N_UNACCOUNT , 0))) || '/' || TO_CHAR(MAX(NVL(N_UNACCOUNT , 0))) ) AS MINMAX_NOTFILES,
             TO_CHAR( TO_CHAR(MIN(NVL(N_INJECTED, 0))) || '/' || TO_CHAR(MAX(NVL(N_INJECTED, 0))) ) AS MINMAX_CLOSED,
             TO_CHAR( TO_CHAR(MIN(NVL(N_NEW, 0))) || '/' || TO_CHAR(MAX(NVL(N_NEW, 0))) ) AS MINMAX_INJECTED,
             TO_CHAR( TO_CHAR(MIN(NVL(N_COPIED, 0))) || '/' || TO_CHAR(MAX(NVL(N_COPIED, 0))) ) AS MINMAX_TRANSFERRED,
             TO_CHAR( TO_CHAR(MIN(NVL(N_CHECKED, 0))) || '/' || TO_CHAR(MAX(NVL(N_CHECKED, 0))) ) AS MINMAX_CHECKED,
             TO_CHAR( TO_CHAR(MIN(NVL(N_DELETED, 0))) || '/' || TO_CHAR(MAX(NVL(N_DELETED, 0))) ) AS MINMAX_DELETED,
             TO_CHAR( INSTANCE_CHECK(RUNNUMBER, MAX(NVL(INSTANCE,0)), MAX(SETUPLABEL), COUNT(NVL(N_CREATED,0)), MAX(NVL(N_INJECTED, 0)) ) ) AS INSTANCE_STATUS,
--           TO_CHAR( NOTFILES_CHECK(RUNNUMBER, MAX(N_UNACCOUNT), MAX(N_INJECTED), MAX(LAST_WRITE_TIME) ) ) AS NOTFILES_STATUS,
             TO_CHAR( NOTFILES_CHECK(RUNNUMBER, MAX(N_UNACCOUNT), MAX(N_INJECTED), MAX(NVL(LAST_WRITE_TIME, '10-APR-10 03.45.00.000000 PM') ) )) AS NOTFILES_STATUS,
             TO_CHAR( GENERATE_FLAG_NOTFILES(RUNNUMBER, MAX(N_UNACCOUNT), MAX(NVL(N_INJECTED, 0)), MAX(LAST_WRITE_TIME)) ||
                      GENERATE_FLAG_CLOSED(RUNNUMBER, MAX(N_INJECTED), MAX(LAST_WRITE_TIME)) ||
                      GENERATE_FLAG_INJECTED(RUNNUMBER, MAX(N_NEW / NVL(N_INJECTED, 1)), MAX(N_NEW) ) ||
                      GENERATE_FLAG_TRANSFERRED(RUNNUMBER, MAX(N_COPIED / NVL(N_NEW, 1)), MAX(N_COPIED) ) ||
                      GENERATE_FLAG_CHECKED(RUNNUMBER, MAX(N_CHECKED / NVL(N_NEW, 1)), MAX(N_CHECKED) ) ||
                      GENERATE_FLAG_DELETED(RUNNUMBER, MIN(START_WRITE_TIME),  MAX(LAST_WRITE_TIME)) ) AS FLAGS,
             TO_CHAR( MAX(run) ) as RANK
FROM (SELECT RUNNUMBER, INSTANCE, SETUPLABEL, N_CREATED, N_INJECTED, N_NEW, N_COPIED, N_CHECKED, N_INSERTED, N_REPACKED, N_DELETED, N_UNACCOUNT, START_WRITE_TIME, LAST_WRITE_TIME, DENSE_RANK() OVER (ORDER BY RUNNUMBER DESC NULLS LAST) run
FROM SM_INSTANCES)
WHERE run <= 30
GROUP BY RUNNUMBER
ORDER BY 1 DESC);

grant select on view_sm_instance_summary to public;
