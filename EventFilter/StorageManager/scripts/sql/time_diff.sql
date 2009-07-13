CREATE OR REPLACE FUNCTION time_diff (
DATE_1 IN DATE, DATE_2 IN DATE) RETURN NUMBER

AS
	result_1   NUMBER;

BEGIN
	select ((DATE_1 - DATE_2) * (86400))
	 into result_1 from dual;

	return result_1;
END time_diff;
/
