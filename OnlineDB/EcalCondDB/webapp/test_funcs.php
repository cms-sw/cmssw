<!--
/* 
 * test_funcs.php
 *
 * Scratch space to see that functions are behaving
 * $Id$
 */
-->

<html>
<body>
<h1>Test Functions</h1>
<pre>

<?php
require_once 'db_functions.php';

$locs = get_loc_list();
var_dump($locs);

$extents = get_run_num_extents();
var_dump($extents);

$extents = get_run_date_extents();
var_dump($extents);

?>
</pre>

<?php
require_once 'index.php';

echo "<form>";
draw_location_box();
draw_runtype_box();
draw_rungentag_box();
draw_run_select_box();
draw_interested_box();
echo "</form>";
?>

</body>
</html>
