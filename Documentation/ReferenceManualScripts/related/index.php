<html>
<head>
<link type="text/css" rel="stylesheet" href="http://cmssdt.cern.ch/SDT/doxygen/doxygen_php_files/doxygen.css">

<style>
.roundbox{
	margin:5px;
	padding:5px;
	-moz-background-clip: border;
	-moz-background-inline-policy: continuous;
	-moz-background-origin: padding;
	-moz-border-radius-bottomleft: 15px;
	-moz-border-radius-bottomright: 15px;
	-moz-border-radius-topleft: 15px;
	-moz-border-radius-topright: 15px;
	-webkit-border-radius: 15px;
	-moz-box-shadow: 0 0 10px rgba(0, 0, 0, 0.4);
	background: #cccccc none repeat scroll 0 0;
	width:115px;	
}

</style>
</head>
<body style="padding:20px; background-color:lightBlue;">
<center>
	<h1>CMSSW Reference Manual (doxygen)</h1>
	<h2>Click on a version </h2>
</center>

<?php

  function getDirectoryList ($directory) 
  {
    $results = array();
    $handler = opendir($directory);
    while ($file = readdir($handler)) {
      if (is_dir($file) && strpos($file, "CMSSW") === 0) {
        $version = explode("_", $file);
	$version_list[$version[1]][$version[2]][$version[3]][] = $file;
      }
    }
    closedir($handler);
    return $version_list;
}

function getDirList()
{

   $output = trim(shell_exec("ls -rs /data/doxygen | grep CMSSW | awk -F \" \" '{print $2}'"));
   $arr = explode("\n", $output);
   
   foreach($arr as $file){
      if (strpos($file, "CMSSW") === 0) {
        $version = explode("_", $file);
	$version_list[$version[1]][$version[2]][$version[3]][] = $file;
      }	
   }

  return $version_list;

}


$BASE = "http://cmssdt.cern.ch/SDT/doxygen/";

//$level1 = getDirectoryList("/data/sdt/SDT/doxygen");
$level1 = getDirList();


krsort($level1);
while (list ($key1, $level2) = each ($level1) ){ 

  krsort($level2);
  while (list ($key2, $level3) = each ($level2) ) { 

    krsort($level3);
    echo "<hr><div class=\"roundbox\"><b>CMSSW_".$key1."_".$key2."_* </b></div>";
    while (list ($key3, $values) = each ($level3) ) { 

      echo "<div class=\"tabs\" style=\"margin-left:150px; width:auto\"><ul class=\"tablist\" style=\"margin:0px;\">";

      sort($values);
      while (list ($key, $value) = each ($values) ) { 
	echo "<li><a target=\"_blank\" href=".$BASE.$value."/doc/html>".$value."</a></li> ";
      }
      echo("</ul></div>");
    }
  }
}

?>
<hr>
<center><a href="mailto:mantas.stankevicius@cern.ch">Contact</a></center>
</body>
</html>