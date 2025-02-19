<?php
/*
 * index.php
 *
 * Run selection page
 * $Id: index.php,v 1.6 2007/04/12 11:04:42 fra Exp $
 */

require_once 'common.php';
require_once 'db_functions.php';

function draw_location_box() {
  $locations = get_loc_list();

  echo "<select name='location'>";
  foreach($locations as $loc) {
    echo "<option value='$loc'>$loc";
  }
  echo "</select>";
}

function draw_sm_box() {
  $supermodules = get_sm_list();
  array_unshift($supermodules, 'Any');

  echo "<select name='SM'>";
  foreach($supermodules as $sm) {
    echo "<option value='$sm'>$sm";
  }
  echo "</select>";
}

/* Returns a list of crystals */
function get_xtal_list() {
  $crystals= array();
  for ($i=1; $i<1701; $i++){
    $crystals[$i]=$i;
 }

  return ($crystals);
}

function draw_xtal_box() {
  $crystals = get_xtal_list();
  array_unshift($crystals, 'Any');

  echo "<select name='CRYSTAL'>";
  foreach($crystals as $xt) {
    echo "<option value='$xt'>$xt";
  }
  echo "</select>";
}

function draw_runtype_box() {
  $runtypes = get_runtype_list();
  array_unshift($runtypes, 'All');
  array_unshift($runtypes, 'All except TEST');

  echo "<select name='run_type'>";
  foreach($runtypes as $type) {
    echo "<option value='$type'>$type";
  }
  echo "</select>";
}

function draw_rungentag_box() {
  $gentags = get_rungentag_list();
  array_unshift($gentags, 'All');

  echo "<select name='run_gen_tag'>";
  foreach($gentags as $tag) {
    echo "<option value='$tag'>$tag";
  }
  echo "</select>";
}

function draw_run_num_range_box() {
  $extents = get_run_num_extents();

  echo "<input type='radio' name='run_select' value='run_range'/> Select by run range:  ";
  echo "<input type='text' name='min_run' value='$extents[MIN_RUN]'>";
  echo " to ";
  echo "<input type='text' name='max_run' value='$extents[MAX_RUN]'>";
}

function draw_run_date_range_box() {
  $extents = get_run_date_extents();
  $min_start = $extents['MIN_START'];
  $max_start = $extents['MAX_START'];
  $fmt = "Y-m-d H:i:s";
  $ary = getdate(strtotime($max_start));

  $def_start = date($fmt, mktime($ary['hours'], $ary['minutes'], $ary['seconds'],
				 $ary['mon'], $ary['mday']-1, $ary['year']));


  echo "<input type='radio' name='run_select' value='date_range'/> Select by date range:  ";
  echo "<input type='text' name='min_start' value='$def_start'>";
  echo " to ";
  echo "<input type='text' name='max_start' value='$max_start'>";
}

function draw_run_select_box() {
  echo "<input type='radio' name='run_select' value='all_runs' checked='checked'/> Select all runs<br/>";
  echo "<input type='radio' name='run_select' value='last_100' /> Select last 100 runs (Auto-Refresh)<br/>";
  draw_run_num_range_box();
  echo "<br/>";
  draw_run_date_range_box();
}

function draw_interested_box() {
  $datatypes = get_datatype_array();
  
  foreach ($datatypes as $name => $prefix) {
    echo "<input type='checkbox' name='$prefix'>$name<br/>";
  }
}

function draw_ordering_box() {
  echo "<select name='run_order'>";
  echo "<option value='desc'>Descending</option>";
  echo "<option value='asc'>Ascending</option>";
  echo "</select>";
}
?>

<!DOCTYPE html
PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html>
<head>
<title>ECAL CondDB Run Selection</title>
<?php echo get_stylelinks(); ?>
</head>

<body>

<h1>ECAL CondDB Run Selection</h1>

<form name='runselect' action='list_runs.php'>
<table class='runselect'>
<tr><th>Location:</th><td><?php draw_location_box(); ?></td></tr>
<tr><th>SM:</th><td><?php draw_sm_box(); ?></td></tr>
<tr><th>Crystal:</th><td><?php draw_xtal_box(); ?></td></tr>
<tr><th>Run Type:</th><td><?php draw_runtype_box(); ?></td></tr>
<tr><th>General Tag:</th><td><?php draw_rungentag_box(); ?></td></tr>
<tr><th>Run Selection:</th><td><?php draw_run_select_box(); ?></td></tr>
<tr><th>Run Order:</th><td><?php draw_ordering_box(); ?></td></tr>
<tr><th>Data:</th><td><?php draw_interested_box(); ?></td></tr>
<!--<tr><td align='left'><input type='checkbox' name='compact'>Compact view</td>-->
<tr><td align='left'><input type='checkbox' name='expanded'>Expanded view</td>
<td colspan='2' align='right'><input type='submit' value='Submit'></td></tr>
</table>
</form>

<p>Send bug reports and feature requests to <a href='mailto:Ricky.Egeland@cern.ch'>Ricky Egeland</a>.</p>

</body>
</html>
