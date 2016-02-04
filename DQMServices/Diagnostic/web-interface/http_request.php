<?php
$method_    =   "tableFromJson";
if (isset ($_GET['tableFromJson'])) 
    $method_=   $_GET['method_'];

$fileName   =   "SiStripFullListgb.txt";
if (isset ($_GET['fileName'])) 
    $fileName=   $_GET['fileName'];

$data=file_get_contents('http://historydqmweb.cern.ch:8181/'.$method_.'?fileName='.$fileName);
echo $data;
?>
