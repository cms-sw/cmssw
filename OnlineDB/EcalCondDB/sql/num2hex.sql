 
CREATE OR REPLACE FUNCTION num2hex (N IN NUMBER) RETURN VARCHAR2 IS
  H  VARCHAR2(64) :='';
  N2 INTEGER      := N;
BEGIN
  LOOP
     SELECT RAWTOHEX(CHR(N2))||H
     INTO   H
     FROM   dual;
 
     N2 := TRUNC(N2 / 256);
     EXIT WHEN N2=0;
  END LOOP;
  RETURN H;
END num2hex;
/
show errors
 