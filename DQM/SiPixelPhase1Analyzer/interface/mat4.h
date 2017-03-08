#ifndef MAT4__H_
#define MAT4__H_

// helper class for matrix operations
// - 4 rows
// - 3 columns
// in math operations behaves like 4x4 matrix with 4th row equal to: [0, 0, 0, 1]
// ! it's just the minimum implementation !
class mat4
{
	public:

	float data[12];
	
	mat4() {}
	
	mat4(float r00, float r10, float r20,
		 float r01, float r11, float r21,
		 float r02, float r12, float r22,
		 float x, float y, float z)
		 {
			 data[0] = r00;
			 data[1] = r10;
			 data[2] = r20;
			 
			 data[3] = r01;
			 data[4] = r11;
			 data[5] = r21;
			 
			 data[6] = r02;
			 data[7] = r12;
			 data[8] = r22;
			 
			 data[9] = x;
			 data[10] = y;
			 data[11] = z;
		 }
	
	mat4(const mat4& mat)
	{
		for (unsigned i = 0; i < 12; ++i) data[i] = mat[i];
	}
	
	mat4& operator&(const mat4& mat)
	{
		if (this != &mat)
		{
			for (unsigned i = 0; i < 12; ++i) data[i] = mat[i];
		}
		return *this;
	}
	
	mat4 operator+(const mat4& mat) const
	{
		mat4 tmp;
		for (unsigned i = 0; i < 12; ++i) tmp[i] = (*this)[i] + mat[i];
		
		return tmp;
	}
	
	mat4 operator*(float s) const
	{
		mat4 tmp;
		for (unsigned i = 0; i < 12; ++i) tmp[i] = (*this)[i] * s;
		
		return tmp;
	}
	
	float& operator[](unsigned i)
	{
		return data[i];
	}
	
	float operator[](unsigned i) const
	{
		return data[i];
	}
	
	void MulVec(const float* vecIn, float* vecOut)
	{
		for (unsigned i = 0; i < 3; ++i)
		{
			float temp = 0;
			for (unsigned j = 0; j < 3; ++j)
			{
				temp += data[3 * j + i] * vecIn[j];
			}
			vecOut[i] = temp + data[9 + i];
		}
	}
	void BuildOrthographicMatrix(float left, float right,
								 float top, float bottom,
								 float near, float far)
	{
		float rmli = 1.0f / (right - left);
		float rpl = right + left;
		
		float tmbi = 1.0f / (top - bottom);
		float tpb = top + bottom;
		
		float fmni = 1.0f / (far - near);
		float fpn = far + near;
		
		data[0] = 2.0f * rmli;
		data[1] = 0.0f;
		data[2] = 0.0f;
		
		data[3] = 0.0f;
		data[4] = 2.0f * tmbi;
		data[5] = 0.0f;
		
		data[6] = 0.0f;
		data[7] = 0.0f;
		data[8] = -2.0f * fmni;
		
		data[9] = -rpl * rmli;
		data[10] = -tpb * tmbi;
		data[11] = -fpn * fmni;
	}
									 
};

#endif