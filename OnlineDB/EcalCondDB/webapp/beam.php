<?php
/* 
 * beam.php
 * 
 * page to display beam data 
 * $Id: 
 */


require_once 'common.php';
require_once 'db_functions.php';

function input_errors() {
  $error = "";
  foreach (array('run_num', 'loc') as $input) {
    if (!isset($input)) { $error .= "<h1>ERROR:  Missing input parameter '$input'</h1>"; }
  }
}
function beam_to_out($run, $loc) {

  $beamresults = fetch_all_beam_data($run, $loc);
  $nbeamrows =  count($beamresults['IOV_ID']);

  if ($nbeamrows ==1) {
  
      foreach($beamresults as $cle=>$valeur)
	{
	  $valeur=$beamresults[$cle][0];
	  
	  echo $cle.' : '.$valeur.'<br>';
	}
     

  } else {
    echo "<tr>
          <th class='typehead'>BEAM</th>
          <td class='noresults'>No BEAM results</td></tr>";
  }

}



// Input
$run = $_GET['run_num'];
$loc = $_GET['loc'];
$title = "Beam data for run $run";



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
  beam_to_out($run, $loc);
}
?>

</body>
</html>
