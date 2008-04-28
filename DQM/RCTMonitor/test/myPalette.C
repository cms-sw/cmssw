void myPalette()
{
  const Int_t colNum = 5;

  Int_t   palette[colNum] = {0};
  Float_t red[colNum]     = {0.};
  Float_t green[colNum]   = {0.};
  Float_t blue[colNum]    = {0.};

  for(Int_t i = 0; i < colNum; i++)
  {
    if(i == 0)
    {
      red[i] = 1.;
    } 

    else if(i == 1)
    {
      red[i]   = 1.;
      green[i] = 0.5;
    }

    else if(i == 2)
    {
      red[i]   = 1.;
      green[i] = 1.;
    }

    else if(i == 3)
    {
      blue[i]  = 1.;
    }

    else if(i == 4)
    {
      green[i] = 1.;
    }

    palette[i] = TColor::GetColor(red[i], green[i], blue[i]);
  }

  gStyle->SetPalette(colNum, palette);
}
