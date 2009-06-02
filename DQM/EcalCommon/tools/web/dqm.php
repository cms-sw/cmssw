<?php

$run = $_REQUEST['run'];

if( file_exists( "/var/www/html/html/" . $run . "/index.html" ) == "TRUE" ) {
  header("location: http://ecalod-web01/html/$run");
}

?>
