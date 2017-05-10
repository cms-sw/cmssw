<?php

function safe($str){

	$str = str_replace('"', "&#34", $str);
	$str = str_replace("'", "&#39", $str);
	$str = str_replace("<", "&lt;", $str);
	$str = str_replace(">", "&gt;", $str);

	return $str;
}

$questionnaire_id = safe($_POST['questionnaire_id']);



try
  { 
  	$db = new PDO('sqlite:aq.db');

switch($_POST['action']){


case "checkpass":

	$password = safe($_POST['pass']);
	$questionnaire_id = safe($_POST['questionnaire_id']);

	$questionnaire = $db->query("SELECT * FROM Questionnaire WHERE id='".$questionnaire_id."'")->fetch();

	if ($questionnaire != null){
		if ($questionnaire['password'] == $password){
			echo "TRUE";
		}
		else{
			echo "FALSE";
		}
	}
	else{
		echo "FALSE";
	}	

break;

case "remindPass":


	$questionnaire = $db->query("SELECT * FROM Questionnaire WHERE id='".$questionnaire_id."'")->fetch();

	if ($questionnaire != null){
		mail($questionnaire['email'], 'Password reminder for '.$questionnaire['publication_number'].' publication', 'Password: '.$questionnaire['password']);
		
		echo "Password sent to ".$questionnaire['email'];
	}
	else{
		echo "Password not sent. Please contact";
	}
	
	$db = NULL;
break;	

}// switch

}
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }


?>