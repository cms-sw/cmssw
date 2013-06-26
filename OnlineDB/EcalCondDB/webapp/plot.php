<?php
/* 
 * plot.php
 * 
 * Plot selection and display page
 * $Id: plot.php,v 1.4 2007/04/12 11:04:42 fra Exp $
 */


require_once 'common.php';
require_once 'db_functions.php';

$conn = connect($_GET['loc']);

function input_errors() {
  $error = "";
  foreach (array('run', 'datatype', 'iov_id', 'exists_str') as $input) {
    if (!isset($input)) { $error .= "<h1>ERROR:  Missing input parameter '$input'</h1>"; }
  }
}

function get_plottypes() {
  return array('histo_all' => 'Histogram (All Channels)',
	       'histo_grp' => 'Histogram (Group Channels)',
	       'graph_all' => 'Graph (All Channels)',
	       'graph_grp' => 'Graph (Group Channels)',
	       'map_all'   => 'Map (All Channels)');
}

function draw_plotselect_form() {
  echo "<div class='plot'>";
  echo "<form name='plotmap' action='plot.php'>";
  echo "Variable:  ";
  draw_plotselect();
  echo "<input type='submit' name='' value='Plot' />";
  // Variable to pass on to next load
  foreach (array('run', 'loc', 'datatype', 'iov_id', 'exists_str') as $name) {
    $value = $_GET[$name];
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
    echo "<select name='plottype'>";
    foreach (get_plottypes() as $value => $label) {
      if (isset($_GET['plottype']) &&
	    $_GET['plottype'] == $value) {
	  $selected = "selected='selected'";
	} else { $selected = ""; }
      echo "<option value='$value' $selected>$label</option>";
    }
    echo "</select>";
  } else {
    echo "No Data Available";
  }
}

function draw_plot() {
  if ( !isset($_GET['tablefield']) ||
       !isset($_GET['iov_id']    ) ||
       !isset($_GET['run'])        ||
       !isset($_GET['plottype'])    ) { return 0; }
  
  echo "<div class='plot'>";
  
  list($table, $field) = split('\.', $_GET['tablefield']);
  $iov_id = $_GET['iov_id'];
  $run = $_GET['run'];
  $plottype = $_GET['plottype'];
  
  $name = "../ecalconddb/plotcache/run$run.$table.$field.$iov_id.$plottype";

  $names = array();
  if ($img_files = glob($name.'*.png')) { // Check cache
    foreach ($img_files as $img) {
      array_push($names, preg_replace('/\.png/', '', $img));
    }
  } elseif ( $names = db_make_rootplot($table, $field,  $iov_id, $plottype, $name) ) { // Draw Plots
  } else { // Error
    echo "<h1>ERROR</h1>";
    $error_msg = get_rootplot_error();
    echo "<p>", $error_msg, "</p>";
  }

  if ($names == 0) {
    echo "Names is zero.";
  }


  foreach ($names as $name) {
    $img = $name.'.png';
    $root = 'download.php?file='.$name.'.root';
    echo "<a href='$root' type='application/octet-stream'><img src=\"$img\" alt=\"$img\"/><br />Click to download ROOT file</a>";
    echo "   Cached on ". date("Y-m-d H:i:s", filemtime($img)). "<br />";
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

<?php
if ($errors = input_errors()) {
  echo $errors;
} else {
  echo "<h1>$title</h1>";
  draw_plotselect_form();
  draw_plot();
}
?>

</body>
</html>
