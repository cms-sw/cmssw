<?php
/* 
 * plot.php
 * 
 * Plot selection and display page
 * $Id$
 */


require_once 'common.php';
require_once 'db_functions.php';

function draw_plotselect_form() {
  echo "<div class='plot'>";
  echo "<form name='plotmap' action='plot.php'>";
  echo "Variable:  ";
  draw_plotselect();
  echo "<input type='submit' name='' value='Plot' />";
  foreach ($_GET as $name => $value) {
    if ($name == "tablefield") { continue; }
    echo "<input type='hidden' name='$name' value='$value' />";
  }
  echo "</form>";
  echo "</div>";
}

function draw_plotselect() {
  global $fieldlists;
  $run = $_GET['run'];
  $datatype = $_GET['datatype'];
  $iov_id = $_GET['iov_id'];
  $exists_str = $_GET['exists_str'];


  $exists = array_flip(split(',', $exists_str));
  $available = array_intersect_key($fieldlists[$datatype], $exists);

  if (count($available)) {
    echo "<select class='plotselect' name='tablefield'>";
    foreach ($available as $table => $tflist) {
      foreach ($tflist as $tf) {
	list($table, $field) = split('\.', $tf);
	$plotparams = db_fetch_plot_params($table, $field);
	if ($plotparams) {
	  $fielddesc = $plotparams['LABEL'];
	} else {
	  $fielddesc = $tf;
	}
	if (isset($_GET['tablefield']) &&
	    $_GET['tablefield'] == $tf) {
	  $selected = "selected='selected'";
	} else { $selected = ""; }
	echo "<option value='$tf' $selected>$fielddesc</option>";
      }
    }
    echo "</select>";
  } else {
    echo "No Data Available";
  }
}

function draw_plot() {
  if ( !isset($_GET['tablefield']) ||
       !isset($_GET['iov_id']    ) ||
       !isset($_GET['run'])      ) { return 0; }
  
  echo "<div class='plot'>";
  
  list($table, $field) = split('\.', $_GET['tablefield']);
  $iov_id = $_GET['iov_id'];
  $run = $_GET['run'];
  
  $plottype = "TH1F";
  $name = "plotcache/run$run.$table.$field.$iov_id.$plottype";

  if ( db_make_rootplot($table, $field,  $iov_id, $plottype, $name) ) {
    $img = $name.'.png';
    $root = 'download.php?file='.$name.'.root';
    echo "<a href='$root' type='application/octet-stream'><img src='$img' /><br />Click to download ROOT file</a>";
  } else { 
    echo "<h1>ERROR</h1>";
    $error_msg = get_rootplot_error();
    echo "<p>", $error_msg, "</p>";
  }

  echo "</div>";
}

// Input
$run = $_GET['run'];
$datatype = $_GET['datatype'];
$iov_id = $_GET['iov_id'];
$exists = $_GET['exists_str'];

$datatypes = get_datatype_array();
$fieldlists = array();
foreach ($datatypes as $name => $prefix) {
  if (isset($_GET[$prefix])) { $ndisplay++; }
  $fieldlists[$prefix] = fetch_field_array(strtoupper($prefix));
}

$dataname = array_search($datatype, $datatypes);
$title = "$dataname plot for run $run";



?>
<!DOCTYPE html
PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html>
<head>
<title><?php echo $title ?></title>
<?php echo get_stylelinks(); ?>
</head>

<body>

<h1><?php echo $title ?></h1>
<?php draw_plotselect_form() ?>
<?php draw_plot() ?>
<pre>
<?php

#echo "INPUT:\n", var_dump($_GET);
?>
</pre>

</body>
</html>