<?php

/*

Backend of `import Country - Institute` from  http://cms.cern.ch/iCMS/jsp/secr/sqlPdb.jsp?type=list
(Frontend tampermonkey javascript script)
@date 2013-07-30
@author Mantas Stankevicius
@email mantas.stankevicius@cern.ch

*/

$country_name = $_POST['country'];
$institute_name = $_POST['institute_name'];
$institute_url = $_POST['institute_url'];


try
  { 
    $db = new PDO('sqlite:aq.db');

    $country = $db->query("SELECT * FROM country WHERE name='".$country_name."' LIMIT 1")->fetchAll();
    if (sizeof($country) == 0){  
    	$db->query("INSERT INTO country (id,name) VALUES (NULL,'".$country_name."')");
	$country_id = $db->lastInsertId();
    }
    else{	
	$country_id = $country[0]['id'];
    }

    if ($institute_name != ""){
	$institute = $db->query("SELECT * FROM institute WHERE name='".$institute_name."' LIMIT 1")->fetchAll();
    	if (sizeof($institute) == 0){    

    		$db->query("INSERT INTO institute (id,name,cadi_url) VALUES (NULL,'".$institute_name."','".$institute_url."')");
		$institute_id = $db->lastInsertId();
    	}
    	else{	
		$institute_id = $institute[0]['id'];
    	}    
    
    	$ci = $db->query("SELECT * FROM country_institute WHERE country_id='".$country_id."' AND institute_id=".$institute_id." LIMIT 1")->fetchAll();
	if (sizeof($ci) == 0){    
	    $db->query("INSERT INTO country_institute (id,country_id,institute_id) VALUES (NULL,'".$country_id."','".$institute_id."')");
	}
    }

    echo "OK";

  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }


?>