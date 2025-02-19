<!--
/* 
 * test_funcs.php
 *
 * Scratch space to see that functions are behaving
 * $Id: test_funcs.php,v 1.5 2007/04/12 11:04:42 fra Exp $
 */
-->

<html>
<body>
<h1>Test Functions</h1>
<pre>

<?php
require_once 'db_functions.php';

echo build_mon_dataset_sql("MON_CRYSTAL_CONSISTENCY_DAT",
			   "task_status = :ts"
			   ), "\n";

$data = fetch_mon_dataset_data("MON_CRYSTAL_CONSISTENCY_DAT", 75, "task_status != 0");
echo " data ";
var_dump($data);

$headers = fetch_mon_dataset_headers("MON_CRYSTAL_CONSISTENCY_DAT");
echo " headers ";
var_dump($headers);

$t_meta = fetch_table_meta("MON_CRYSTAL_CONSISTENCY_DAT");
echo " table meta ";
var_dump($t_meta);

$f_meta = fetch_field_meta("MON_CRYSTAL_CONSISTENCY_DAT");
echo " field meta ";
var_dump($f_meta);

$c_meta = fetch_channel_meta("EB_crystal_number");
echo " channel meta ";
var_dump($c_meta);

?>
</pre>


</body>
</html>
